import os
import torch
import copy
import json
import argparse
import logging
import torch.nn as nn
import torch.optim as optim
import gc
from typing import List, Tuple
from torch_geometric.data import Batch
from arguments import parse_arguments
from model.agent import GRPOAgent
from model.decoder import DecoderState
from model.utils import mol_graph2smiles
from model.reward_model import RewardModel
from model.mol_graph import MolGraph
from model.vocab import MotifVocab
from model.encoder import Encoder, Atom_Embedding
from model.nn import GIN_virtual
from model.MiCaM_VAE import MiCaM, GeneratorFromModel
from model.mydataclass import ModelParams
from model.benchmarks import QuickBenchmark


def clone_state(state: DecoderState) -> DecoderState:
    new_state = DecoderState(
        latent_repr=state.latent_repr.detach().clone(),  # 保证不含计算图
        batch_idx=state.batch_idx,
        return_trace=(state.trace is not None)
    )
    new_state.current_graph = state.current_graph.copy()
    new_state.atoms_step_dict = state.atoms_step_dict.copy()
    new_state.connections_list = state.connections_list.copy()
    new_state.decode_step = state.decode_step
    new_state.query_atom = state.query_atom
    new_state.query_bond_type = state.query_bond_type
    new_state.current_graph_data = state.current_graph_data  # 小心：这里你也可以 .clone() 其 x 属性
    new_state.motif_node_vecs = state.motif_node_vecs
    new_state.motif_graph_vecs = state.motif_graph_vecs
    return new_state

def compute_grpo_loss_strict(agent, ref_agent,
                             states_batch: List[List[DecoderState]],
                             actions_batch: List[List[Tuple]],
                             log_probs_old_batch: List[List[torch.Tensor]],
                             rewards: List[float],
                             beta: float,
                             eps_clip: float = 0.2) -> torch.Tensor:
    """
    严格按照 GRPO 论文公式计算 Loss，支持 group-wise reward 标准化与轨迹级 KL。
    """
    G = len(states_batch)
    device = next(agent.decoder.parameters()).device

    # Step 1: 轨迹级 reward 标准化（group-wise）
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    reward_mean = rewards_tensor.mean()
    reward_std = rewards_tensor.std() + 1e-6
    advantages = (rewards_tensor - reward_mean) / reward_std

    losses = []  # ✅ 用来收集每个 trajectory 的 loss

    for i in range(G):
        states = states_batch[i]
        actions = actions_batch[i]
        log_probs_old = log_probs_old_batch[i]
        if len(actions) == 0:
            continue

        A = advantages[i]
        traj_loss = 0.0
        log_p_sum = 0.0
        log_p_ref_sum = 0.0

        for t, (state, action) in enumerate(zip(states, actions)):
            # 当前策略 log_prob（有梯度）
            log_prob = agent.get_log_prob(state, action)
            # 旧策略 log_prob（无梯度）
            log_prob_old = log_probs_old[t]
            # 参考策略 log_prob（无梯度）
            with torch.no_grad():
                log_prob_ref = ref_agent.get_log_prob(state, action)

            # Clipped PPO-style surrogate loss
            ratio = torch.exp(log_prob - log_prob_old)
            clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip)
            loss_token = -torch.min(ratio * A, clipped_ratio * A)
            traj_loss += loss_token

            log_p_sum += log_prob
            log_p_ref_sum += log_prob_ref

        traj_loss = traj_loss / len(actions)

        # KL 估计
        log_ratio = log_p_ref_sum - log_p_sum
        kl_traj = (torch.exp(log_ratio) - log_ratio - 1) / len(actions)

        loss_i = traj_loss + beta * kl_traj  # ✅ 单条轨迹 loss
        losses.append(loss_i)

        if i % 50 == 0:
            print(f"[GRPO Loss Breakdown i={i}] "
                  f"Surrogate Loss: {traj_loss.item():.4f}, "
                  f"KL: {kl_traj.item():.4f}, "
                  f"Total: {loss_i.item():.4f}")
            print(
                f"[Advantage] A: {A.item():.4f} | Reward: {rewards[i]:.4f} | mean={reward_mean.item():.4f}, std={reward_std.item():.4f}")

    if len(losses) == 0:
        print("[Warning] GRPO Loss skipped: no valid trajectories in batch.")
        return torch.tensor(0.0, device=device, requires_grad=True)
    # ✅ 返回 mean loss（保留梯度）
    return torch.stack(losses).mean()


class GRPOTrainer:
    def __init__(self, agent, ref_agent, reward_model, args, model):
        self.agent = agent
        self.ref_agent = ref_agent
        self.reward_model = reward_model
        self.args = args
        self.model = model
        self.save_path = args.save_path
        self.beta = args.beta

        # 设置优化器：训练 decoder + motif_encoder + graph_encoder
        params_to_optimize = list(model.decoder.parameters())
        self.optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)

        self.best_reward = -float("inf")
        self.agent.decoder.train()
        self.model.decoder.train()
        self.ref_agent.decoder.eval()  # frozen reference policy

    def sample_trajectories(self, z, num_per_z, max_decode_step=20):
        z = z.to(next(self.agent.decoder.parameters()).device)

        trajectories = []
        states_batch = []
        actions_batch = []
        log_probs_batch = []
        motif_lists = []

        motif_graphs = Batch.from_data_list(self.agent.decoder.motif_graphs.to_data_list()).to(z.device)
        motif_node_vecs_all, motif_graph_vecs = self.agent.decoder.motif_encoder(motif_graphs)
        motif_node_vecs = motif_node_vecs_all[self.agent.decoder.motif_vocab.get_conns_idx()]

        for i in range(z.shape[0]):
            for _ in range(num_per_z):
                state = DecoderState(latent_repr=z[i], batch_idx=i)

                # 使用策略网络 startNN + softmax 分布进行采样
                latent = z[i].unsqueeze(0)  # shape: [1, latent_size]
                start_scores = torch.matmul(self.agent.decoder.startNN(latent), motif_graph_vecs.T)  # [1, N_motifs]
                start_probs = torch.softmax(start_scores, dim=-1).squeeze(0)  # [N_motifs]
                motif_idx = torch.multinomial(start_probs, num_samples=1).item()
                motif = self.agent.decoder.motif_list[motif_idx]

                state.add_motif(motif)

                states_i = []
                actions_i = []
                log_probs_i = []
                motif_trace = [motif]

                while state.non_terminal and state.decode_step < max_decode_step:
                    action, log_prob = agent.sample(state, motif_node_vecs=motif_node_vecs, motif_graph_vecs=motif_graph_vecs)
                    states_i.append(clone_state(state))
                    actions_i.append(action)
                    log_probs_i.append(log_prob.detach())  # ⬅️ detach 防止影响计算图

                    if action[0] == 'cyc':
                        state.cyclize(action[1])
                    else:
                        motif = self.agent.decoder.motif_list[action[0]]
                        state.add_motif(motif, connection_order=action[1])
                        motif_trace.append(motif)

                if state.decode_step == 0:
                    continue  # 🚨 这条轨迹没有拼接任何 motif，跳过

                trajectories.append(mol_graph2smiles(state.current_graph))
                states_batch.append(states_i)
                actions_batch.append(actions_i)
                log_probs_batch.append(log_probs_i)
                motif_lists.append(motif_trace)

                del state.current_graph_data
                del state.motif_node_vecs
                del state.motif_graph_vecs
                del states_i, actions_i, log_probs_i, state
                torch.cuda.empty_cache()
                gc.collect()

        return trajectories, states_batch, actions_batch, log_probs_batch, motif_lists

    def benchmark(self, train_path, greedy):
        generator = GeneratorFromModel(self.model, greedy=greedy)
        train_set = [s.strip() for s in open(train_path)]
        return QuickBenchmark(training_set=train_set, num_samples=10000).assess_model(generator)

    def train(self, num_steps, batch_size, latent_size):
        G = self.args.num_trajectories_per_z
        for step in range(num_steps):
            z = torch.randn(batch_size, latent_size, device=next(self.agent.decoder.parameters()).device)
            z = z.repeat_interleave(G, dim=0)
            trajectories, states, actions, log_probs_old_batch, motif_lists = self.sample_trajectories(z, num_per_z=1)
            rewards = self.reward_model.evaluate(trajectories, motif_lists)
            loss = compute_grpo_loss_strict(self.agent, self.ref_agent, states, actions, log_probs_old_batch, rewards, self.beta)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            import gc
            gc.collect()

            reward_mean = torch.tensor(rewards).mean().item()
            avg_len = sum(len(x) for x in actions) / len(actions)
            logging.info(f"Step {step} | Loss: {loss.item():.4f} | Reward Mean: {reward_mean:.4f} | Avg Steps: {avg_len:.2f}")

            if reward_mean > self.best_reward:
                self.best_reward = reward_mean
                torch.save(self.model.state_dict(), self.save_path)
                logging.info(f"💾 New best model saved! Mean reward: {reward_mean:.4f}")

        logging.info("📊 Benchmarking best model...")
        with torch.no_grad():
            state_dict = torch.load(self.save_path)
            self.agent.decoder.load_state_dict(state_dict, strict=False)
            self.model.load_state_dict(state_dict, strict=False)
            results = self.benchmark(train_path=self.args.train_path, greedy=self.args.greedy)
            logging.info(results)


if __name__ == "__main__":
    args = parse_arguments()
    args.final_score_weights = json.loads(args.final_score_weights)
    torch.cuda.set_device(args.cuda)

    log_path = os.path.join(os.path.dirname(args.save_path), "grpo_train.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logging.info("🚀 Start GRPO training for MiCaM+GRPO (decoder + encoder trainable)...")

    MolGraph.load_operations(args.operation_path)
    MolGraph.load_vocab(args.vocab_path)

    model_params = ModelParams(args)
    model = MiCaM(model_params).cuda()

    # ✅ 加载训练好的 decoder + encoder 参数
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt[0], strict=False)
    model.load_motifs_embed(args.motif_embed_path)

    agent = GRPOAgent(model.decoder)
    ref_agent = GRPOAgent(copy.deepcopy(model.decoder))
    for p in ref_agent.decoder.parameters():
        p.requires_grad = False

    reward_model = RewardModel(weights=args.final_score_weights, cooccur_path=args.cooccur_path)
    trainer = GRPOTrainer(agent, ref_agent, reward_model, args, model)
    trainer.train(num_steps=args.num_steps, batch_size=args.batch_size, latent_size=args.latent_size)
