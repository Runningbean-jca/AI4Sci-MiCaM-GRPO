import os
import torch
import copy
import json
import argparse
import logging
import torch.nn as nn
import torch.optim as optim

from typing import List, Tuple
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


def compute_grpo_loss_strict(agent, ref_agent,
                             states_batch: List[List[DecoderState]],
                             actions_batch: List[List[Tuple]],
                             rewards: List[float],
                             beta: float) -> torch.Tensor:
    G = len(states_batch)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=next(agent.decoder.parameters()).device)
    norm_rewards = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-6)

    total_loss = 0.0
    for i in range(G):
        states, actions = states_batch[i], actions_batch[i]
        if len(actions) == 0:
            continue
        A = norm_rewards[i]
        traj_loss = 0.0
        for state, action in zip(states, actions):
            log_prob = agent.get_log_prob(state, action)
            log_prob_ref = ref_agent.get_log_prob(state, action)

            log_ratio = log_prob - log_prob_ref
            ratio = torch.exp(log_ratio)

            # ✅ clip ratio
            eps = 0.2
            clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)
            loss_token = -torch.min(ratio * A, clipped_ratio * A)

            traj_loss += loss_token

        traj_loss /= len(actions)
        total_loss += traj_loss

    return total_loss / G


class GRPOTrainer:
    def __init__(self, agent, ref_agent, reward_model, args, model):
        self.agent = agent
        self.ref_agent = ref_agent
        self.reward_model = reward_model
        self.args = args
        self.model = model
        self.save_path = args.save_path
        self.beta = args.beta

        # ✅ decoder + motif encoder 都参与训练
        self.optimizer = torch.optim.Adam(
            agent.decoder.parameters(),
            lr=args.lr
        )

        self.best_reward = -float("inf")
        self.agent.decoder.eval()
        self.ref_agent.decoder.eval()
        self.model.decoder.eval()

    def sample_trajectories(self, z, num_per_z, max_decode_step=20):
        z = z.to(next(self.agent.decoder.parameters()).device)
        trajectories, states_batch, actions_batch, motif_lists = [], [], [], []
        for i in range(z.shape[0]):
            for _ in range(num_per_z):
                state = DecoderState(latent_repr=z[i], batch_idx=i)
                motif = self.agent.decoder.motif_list[torch.randint(len(self.agent.decoder.motif_list), (1,)).item()]
                state.add_motif(motif)

                states_i, actions_i, motif_trace = [], [], [motif]
                while state.non_terminal and state.decode_step < max_decode_step:
                    action, _ = self.agent.sample(state)
                    states_i.append(copy.deepcopy(state))
                    actions_i.append(action)
                    if action[0] == 'cyc':
                        state.cyclize(action[1])
                    else:
                        motif = self.agent.decoder.motif_list[action[0]]
                        state.add_motif(motif, connection_order=action[1])
                        motif_trace.append(motif)

                states_batch.append(states_i)
                actions_batch.append(actions_i)
                motif_lists.append(motif_trace)
                trajectories.append(mol_graph2smiles(state.current_graph))
        return trajectories, states_batch, actions_batch, motif_lists

    def benchmark(self, train_path, greedy):
        generator = GeneratorFromModel(self.model, greedy=greedy)
        train_set = [s.strip() for s in open(train_path)]
        return QuickBenchmark(training_set=train_set, num_samples=10000).assess_model(generator)

    def train(self, num_steps, batch_size, latent_size):
        G = self.args.num_trajectories_per_z
        for step in range(num_steps):
            z = torch.randn(batch_size, latent_size, device=next(self.agent.decoder.parameters()).device)
            z = z.repeat_interleave(G, dim=0)
            trajectories, states, actions, motif_lists = self.sample_trajectories(z, num_per_z=1)
            rewards = self.reward_model.evaluate(trajectories, motif_lists)
            loss = compute_grpo_loss_strict(self.agent, self.ref_agent, states, actions, rewards, self.beta)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            reward_mean = torch.tensor(rewards).mean().item()
            avg_len = sum(len(x) for x in actions) / len(actions)
            logging.info(f"Step {step} | Loss: {loss.item():.4f} | Reward Mean: {reward_mean:.4f} | Avg Steps: {avg_len:.2f}")

            if reward_mean > self.best_reward:
                self.best_reward = reward_mean
                torch.save(self.agent.decoder.state_dict(), self.save_path)
                logging.info(f"💾 New best model saved! Mean reward: {reward_mean:.4f}")

        logging.info("📊 Benchmarking best model...")
        with torch.no_grad():
            state_dict = torch.load(self.save_path)
            self.agent.decoder.load_state_dict(state_dict, strict=False)
            self.model.decoder.load_state_dict(state_dict, strict=False)
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
