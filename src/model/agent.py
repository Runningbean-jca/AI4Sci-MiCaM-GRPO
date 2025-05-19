import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

class GRPOAgent:
    def __init__(self, decoder):
        """
        初始化 GRPO Agent。
        decoder 是训练好的 MiCaM.decoder 模块，包含 startNN / queryNN / keyNN / motif_encoder。
        """
        self.decoder = decoder

    def batched_keyNN(self, tensor, batch_size=512):
        results = []
        for i in range(0, tensor.size(0), batch_size):
            results.append(self.decoder.keyNN(tensor[i:i + batch_size]))
        return torch.cat(results, dim=0)

    def sample(self, state):
        """
        从当前状态 sample 一个动作并返回其 log_prob。
        """
        graph_data = state.current_graph_data
        z = state.latent_repr.unsqueeze(0)
        query_atom = state.query_atom
        bond_type = state.query_bond_type
        device = z.device

        # 图编码
        node_vecs, graph_vecs = self.decoder.graph_encoder(graph_data.to(device))

        # motif encoder 动态编码
        motif_graphs = Batch.from_data_list(self.decoder.motif_graphs.to_data_list()).to(device)
        motif_node_vecs, motif_graph_vecs = self.decoder.motif_encoder(motif_graphs)
        motif_node_vecs = motif_node_vecs[self.decoder.motif_vocab.get_conns_idx()]
        state.motif_node_vecs = motif_node_vecs.detach()
        state.motif_graph_vecs = motif_graph_vecs.detach()

        if state.decode_step == 0:
            logits = torch.matmul(self.decoder.startNN(z), motif_graph_vecs.T)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            motif_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(motif_idx, device=device))
            return (motif_idx, None), log_prob

        # t > 0：构造 query 向量
        query_vec = self.decoder.queryNN(
            torch.cat([z, graph_vecs, node_vecs[query_atom].unsqueeze(0)], dim=-1)
        )

        conn_indices = self.decoder.motif_vocab.bond_type_conns_dict[bond_type]
        motif_node_vecs = state.motif_node_vecs
        motif_conn_vecs = self.batched_keyNN(motif_node_vecs[conn_indices])
        scores_motif = torch.matmul(query_vec, motif_conn_vecs.T).squeeze(0)

        # 闭环候选
        cyc_cands, cyc_vecs = [], []
        for cyc in state.connections_list[1:]:
            if state.current_graph.nodes[cyc]['dummy_bond_type'] == bond_type and \
                    state.atoms_step_dict[cyc] != state.atoms_step_dict[query_atom]:
                cyc_cands.append(cyc)
                cyc_vecs.append(node_vecs[cyc])

        if cyc_vecs:
            cyc_vecs = torch.stack(cyc_vecs)
            cyc_vecs = self.decoder.keyNN(cyc_vecs)
            scores_cyc = torch.matmul(query_vec, cyc_vecs.T).squeeze(0)
            scores = torch.cat([scores_motif, scores_cyc], dim=-1)
        else:
            scores = scores_motif

        probs = torch.softmax(scores, dim=-1)
        dist = torch.distributions.Categorical(probs)
        sampled_idx = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(sampled_idx, device=device))

        if sampled_idx < len(conn_indices):
            motif_idx, conn_order = self.decoder.motif_vocab.from_conn_idx(conn_indices[sampled_idx])
            action = (motif_idx, conn_order)
        else:
            cyc_idx = sampled_idx - len(conn_indices)
            action = ("cyc", cyc_cands[cyc_idx])

        return action, log_prob

    def get_log_prob(self, state, action):
        """
        给定已执行的动作，返回 log π(a|s)。
        """
        graph_data = state.current_graph_data
        z = state.latent_repr.unsqueeze(0)
        query_atom = state.query_atom
        bond_type = state.query_bond_type
        device = z.device

        node_vecs, graph_vecs = self.decoder.graph_encoder(graph_data.to(device))
        query_vec = self.decoder.queryNN(
            torch.cat([z, graph_vecs, node_vecs[query_atom].unsqueeze(0)], dim=-1)
        )

        conn_indices = self.decoder.motif_vocab.bond_type_conns_dict[bond_type]
        motif_node_vecs = state.motif_node_vecs
        motif_conn_vecs = self.batched_keyNN(motif_node_vecs[conn_indices])
        scores_motif = torch.matmul(query_vec, motif_conn_vecs.T).squeeze(0)

        cyc_cands, cyc_vecs = [], []
        for cyc in state.connections_list[1:]:
            if state.current_graph.nodes[cyc]['dummy_bond_type'] == bond_type and \
                    state.atoms_step_dict[cyc] != state.atoms_step_dict[query_atom]:
                cyc_cands.append(cyc)
                cyc_vecs.append(node_vecs[cyc])

        if cyc_vecs:
            cyc_vecs = torch.stack(cyc_vecs)
            cyc_vecs = self.decoder.keyNN(cyc_vecs)
            scores_cyc = torch.matmul(query_vec, cyc_vecs.T).squeeze(0)
            scores = torch.cat([scores_motif, scores_cyc], dim=-1)
        else:
            scores = scores_motif

        probs = torch.softmax(scores, dim=-1)

        if action[0] == "cyc":
            idx = cyc_cands.index(action[1])
            final_idx = len(conn_indices) + idx
        else:
            for i, conn_idx in enumerate(conn_indices):
                motif_idx, order = self.decoder.motif_vocab.from_conn_idx(conn_idx)
                if motif_idx == action[0] and order == action[1]:
                    final_idx = i
                    break

        return torch.log(probs[final_idx] + 1e-8)






