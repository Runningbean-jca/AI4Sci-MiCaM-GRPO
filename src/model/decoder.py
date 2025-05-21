import torch
import torch.nn as nn
from rdkit import Chem
from model.mol_graph import MolGraph, BOND_VOCAB
from model.encoder import Encoder
import networkx as nx
from model.utils import get_accuracy, mol_graph2smiles, networkx2data, sample_from_distribution, graph2mol
from typing import Dict, List, Tuple, Optional, Union
from model.vocab import MotifVocab
from torch_geometric.data import Data, Batch
from model.nn import MLP
from model.mydataclass import batch_train_data, Decoder_Output


class DecoderState(object):
    def __init__(
            self,
            latent_repr: torch.Tensor,
            batch_idx: int,
            return_trace: bool = False
    ):
        self.latent_repr = latent_repr
        self.batch_idx = batch_idx

        self.current_graph = nx.Graph()
        self.current_graph_data = None
        self.atoms_step_dict: Dict[int, int] = {}
        self.connections_list: List[int] = []
        self.trace: Union[List[str], None] = [] if return_trace else None
        self.non_terminal: bool = True

        self.query_atom: int = None
        self.query_bond_type = None

        self.decode_step: int = 0
        self.motif_node_vecs = None  # 新增：缓存节点向量
        self.motif_graph_vecs = None  # 新增：缓存图向量

    def merge_atoms(self, atom1: int, atom2: int) -> None:
        graph = self.current_graph
        self.connections_list.remove(atom1)
        self.connections_list.remove(atom2)
        self.atoms_step_dict.pop(atom1)
        self.atoms_step_dict.pop(atom2)
        assert graph.nodes[atom1]['dummy_bond_type'] == graph.nodes[atom2]['dummy_bond_type']
        bond_type = graph.nodes[atom1]['dummy_bond_type']
        anchor1 = list(graph.neighbors(atom1))[0]
        anchor2 = list(graph.neighbors(atom2))[0]
        graph.add_edge(anchor1, anchor2)
        graph[anchor1][anchor2]['bondtype'] = bond_type
        graph[anchor1][anchor2]['label'] = BOND_VOCAB[bond_type]
        graph.remove_node(atom1)
        graph.remove_node(atom2)

        mapping = dict(zip(graph.nodes(), range(0, len(graph))))
        self.current_graph = nx.relabel_nodes(graph, mapping)
        self.connections_list = [mapping[i] for i in self.connections_list]
        new_step_dict = dict()
        for key, value in self.atoms_step_dict.items():
            new_step_dict[mapping[key]] = value
        self.atoms_step_dict = new_step_dict

    def state_update(self) -> None:
        if len(self.connections_list) == 0:
            self.non_terminal = False
        else:
            self.query_atom = self.connections_list[0]
            self.query_bond_type = self.current_graph.nodes[self.query_atom]['dummy_bond_type']
        self.current_graph_data, _ = networkx2data(self.current_graph)
        if self.trace is not None:
            self.trace.append(mol_graph2smiles(self.current_graph, postprocessing=False))

    def add_motif(self, motif_smiles: str, connection_order: Optional[int] = None) -> None:
        self.decode_step += 1

        motif_graph, new_conn_list, ranks = MolGraph.motif_to_graph(motif_smiles)
        if connection_order is not None:
            for i in range(len(motif_graph)):
                if ranks[i] == connection_order:
                    connection_idx = i + len(self.current_graph)
                    break
        motif_graph = nx.convert_node_labels_to_integers(motif_graph, len(self.current_graph))

        for i, conn in enumerate(new_conn_list):
            conn = conn + len(self.current_graph)
            new_conn_list[i] = conn
            self.atoms_step_dict[conn] = self.decode_step

        self.connections_list = self.connections_list + new_conn_list
        self.current_graph = nx.union(self.current_graph, motif_graph)

        if connection_order is not None:
            self.merge_atoms(self.connections_list[0], connection_idx)
        self.state_update()

    def cyclize(self, connection_idx) -> None:
        self.decode_step += 1
        self.merge_atoms(self.query_atom, connection_idx)
        self.state_update()

    def result(self, return_trace: bool = False) -> Union[List[str], List[Tuple[str, List[str]]]]:
        for atom in list(self.current_graph.nodes):
            if self.current_graph.nodes[atom]['smarts'] == '*':
                self.current_graph.nodes[atom]['smarts'] = 'C'
        smiles = mol_graph2smiles(self.current_graph)
        if self.trace:
            self.trace.append(smiles)
        return (smiles, self.trace) if return_trace else smiles

    def to_mol(self) -> Optional[Chem.Mol]:
        """
        将当前构建图转为 RDKit Mol 对象（用于化学规则检查）
        """
        try:
            mol = graph2mol(self.current_graph)
            return mol
        except Exception as e:
            print(f"[DecoderState.to_mol] Graph 转换失败: {e}")
            return None


class Decoder(nn.Module):
    def __init__(self,
                 atom_embedding: nn.Module,
                 edge_embedding: nn.Module,
                 decoder_gnn: nn.Module,
                 motif_gnn: nn.Module,
                 motif_vocab: MotifVocab,
                 motif_graphs: Batch,
                 motif_embed_size: int,
                 hidden_size: int,
                 latent_size: int,
                 dropout: float,
                 use_motif_embed: bool = True
                 ):
        super(Decoder, self).__init__()
        self.motif_vocab = motif_vocab
        self.motif_list = motif_vocab.motif_smiles_list

        self.motif_embed_size = motif_embed_size
        self.hidden_size = hidden_size

        self.motif_graphs = motif_graphs
        self.graph_encoder = Encoder(atom_embedding, edge_embedding, decoder_gnn)
        self.motif_encoder = Encoder(atom_embedding, edge_embedding, motif_gnn)

        # 不再保留静态 embedding，全部改为动态计算
        self.motif_node_embed = None  # 删除静态 embedding
        self.motif_graph_embed = None

        self.startNN = MLP(
            in_channels=latent_size,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            num_layers=3,
            act=nn.ReLU(inplace=True),
            dropout=dropout,
        )

        self.queryNN = MLP(
            in_channels=latent_size + 2 * hidden_size,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            num_layers=3,
            act=nn.ReLU(inplace=True),
            dropout=dropout,
        )

        self.keyNN = MLP(
            in_channels=hidden_size,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            num_layers=3,
            act=nn.ReLU(inplace=True),
            dropout=dropout,
        )

        self.start_loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
        self.query_loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)

    def forward(
            self,
            z: torch.Tensor,
            input: batch_train_data,
            dev: bool = False,
    ):

        batch_size = len(z)

        batch_motif_graphs = self.motif_graphs.index_select(input.motifs_list)
        batch_motif_graphs = Batch.from_data_list(batch_motif_graphs).to(z.device)
        motif_node_vecs, motif_graph_vecs = self.motif_encoder(batch_motif_graphs)

        start_scores = torch.matmul(self.startNN(z), motif_graph_vecs.T)
        start_loss = self.start_loss(start_scores, input.batch_start_labels) / batch_size
        with torch.no_grad():
            tart_acc, start_topk_acc = get_accuracy(start_scores, input.batch_start_labels)

        node_vecs, graph_vecs = self.graph_encoder(input.batch_train_graphs)

        query = self.queryNN(torch.cat((
            z[input.mol_idx], graph_vecs[input.graph_idx], node_vecs[input.query_idx]), dim=-1))

        if dev:
            key = self.keyNN(torch.cat((
                motif_node_vecs, node_vecs[input.cyclize_cand_idx]), dim=0))
        else:
            key = self.keyNN(torch.cat((
                motif_node_vecs[input.motif_conns_idx], node_vecs[input.cyclize_cand_idx]), dim=0))

        query_scores = torch.matmul(query, key.T)
        query_loss = self.query_loss(query_scores, input.labels) / batch_size
        with torch.no_grad():
            query_acc, query_topk_acc = get_accuracy(query_scores, input.labels)

        decoder_loss = start_loss + query_loss

        return Decoder_Output(
            decoder_loss=decoder_loss,
            start_loss=start_loss,
            query_loss=query_loss,
            tart_acc=tart_acc,
            start_topk_acc=start_topk_acc,
            query_acc=query_acc,
            query_topk_acc=query_topk_acc,
        )

    def save_motifs_embed(self, file):
        motif_node_embed, motif_graph_embed = torch.Tensor([]), torch.Tensor([])
        motif_graphs_list = self.motif_graphs.to_data_list()
        for b in range((len(motif_graphs_list) - 1) // 2000 + 1):
            motif_graphs = Batch.from_data_list(motif_graphs_list[b * 2000: (b + 1) * 2000]).cuda()
            node_embed, graph_embed = self.motif_encoder(motif_graphs)
            motif_node_embed = torch.cat((motif_node_embed, node_embed.cpu()), dim=0)
            motif_graph_embed = torch.cat((motif_graph_embed, graph_embed.cpu()), dim=0)
            torch.cuda.empty_cache()
        motif_node_embed = motif_node_embed[self.motif_vocab.get_conns_idx()]
        motif_embed = (motif_node_embed, motif_graph_embed)
        torch.save((motif_node_embed, motif_graph_embed), file)

    def load_motifs_embed(self, file):
        node_embed, graph_embed = torch.load(file)
        self.motif_node_embed = nn.Parameter(node_embed.cuda())
        self.motif_graph_embed = nn.Parameter(graph_embed.cuda())

    def pick_fisrt_motifs_for_batch(
            self,
            latent_reprs: torch.Tensor,
            decoder_states: List[DecoderState],
            greedy: bool,
            beam_top: int,
            temperature: float,
    ) -> List[DecoderState]:

        # start_scores = torch.softmax(temperature * torch.matmul(self.startNN(latent_reprs), self.motif_graph_embed.T),
        #                              dim=-1)
        motif_graphs = Batch.from_data_list(self.motif_graphs.to_data_list()).to(self.device)
        _, motif_graph_vecs = self.motif_encoder(motif_graphs)
        start_scores = torch.softmax(temperature * torch.matmul(self.startNN(latent_reprs), motif_graph_vecs.T), dim=-1)
        motif_indices = sample_from_distribution(start_scores, greedy=greedy, topk=beam_top)

        for decoder_state in decoder_states:
            motif_idx = motif_indices[decoder_state.batch_idx]
            decoder_state.add_motif(motif_smiles=self.motif_list[motif_idx])

        return decoder_states

    def connection_query_step(
            self,
            query: torch.Tensor,
            decoder_state: DecoderState,
            scores_motif_connection: torch.Tensor,
            graph_data: Data,
            greedy: bool,
            beam_top: int,
            temperature: float,
    ):
        bond_type = decoder_state.query_bond_type

        cyc_cands, cyc_vecs = [], []
        for cyc_cand in decoder_state.connections_list[1:]:
            if decoder_state.atoms_step_dict[decoder_state.query_atom] == decoder_state.atoms_step_dict[cyc_cand] or \
                    decoder_state.current_graph.nodes[cyc_cand]['dummy_bond_type'] != decoder_state.query_bond_type:
                continue
            cyc_cands.append(cyc_cand)
            cyc_vecs.append(graph_data.x[cyc_cand])
        if len(cyc_vecs) > 0:
            cyc_vecs = torch.stack(cyc_vecs)
            key = self.keyNN(cyc_vecs)
            scores_cyc_cands = torch.matmul(query, key.T)

            scores = torch.cat(
                (scores_motif_connection[self.motif_vocab.bond_type_conns_dict[bond_type]], scores_cyc_cands), dim=-1)

        else:
            scores = scores_motif_connection[self.motif_vocab.bond_type_conns_dict[bond_type]]

        distr = torch.softmax(temperature * scores, dim=-1)
        idx = sample_from_distribution(distr, greedy=greedy, topk=beam_top)

        if idx < len(self.motif_vocab.bond_type_conns_dict[bond_type]):
            ## add motif phase
            motif_idx, connection_order = self.motif_vocab.from_conn_idx(
                self.motif_vocab.bond_type_conns_dict[bond_type][idx])
            decoder_state.add_motif(self.motif_list[motif_idx], connection_order=connection_order)
        else:
            ## cyclize phase
            idx = idx - len(self.motif_vocab.bond_type_conns_dict[bond_type])
            decoder_state.cyclize(cyc_cands[idx])

    def connetion_query_steps_for_batch(
            self,
            decoder_states: List[DecoderState],
            max_decode_step: int,
            greedy: bool,
            beam_top: int,
            temperature: float,
    ) -> List[DecoderState]:

        for _ in range(1, max_decode_step + 1):
            nonterm_states: List[DecoderState] = [decoder_state for decoder_state in decoder_states if
                                                  decoder_state.non_terminal]

            if len(nonterm_states) == 0:
                break

            batch_graph_data = Batch.from_data_list(
                [decoder_state.current_graph_data for decoder_state in nonterm_states]).to(self.device)
            batch_graph_data.x, graph_reprs = self.graph_encoder(batch_graph_data)
            batch_graph_data = Batch.to_data_list(batch_graph_data)

            latent_reprs = torch.stack([decoder_state.latent_repr for decoder_state in nonterm_states])
            query_atoms_reprs = torch.stack(
                [batch_graph_data[idx].x[decoder_state.connections_list[0]] for idx, decoder_state in
                 enumerate(nonterm_states)])
            query = self.queryNN(torch.cat((latent_reprs, graph_reprs, query_atoms_reprs), -1))

            # key_motif_connections = self.keyNN(self.motif_node_embed)
            # node_embed, _ = self.motif_encoder(motif_graphs)
            # node_embed = node_embed[self.motif_vocab.get_conns_idx()]
            motif_graphs = Batch.from_data_list(self.motif_graphs.to_data_list()).to(self.device)
            node_embed, _ = self.motif_encoder(motif_graphs)
            node_embed = node_embed[self.motif_vocab.get_conns_idx()]
            key_motif_connections = self.keyNN(node_embed)
            scores_motif_connection = torch.matmul(query, key_motif_connections.T)

            for idx, decoder_state in enumerate(nonterm_states):
                self.connection_query_step(
                    decoder_state=decoder_state,
                    query=query[idx],
                    scores_motif_connection=scores_motif_connection[idx],
                    graph_data=batch_graph_data[idx],
                    greedy=greedy,
                    beam_top=beam_top,
                    temperature=temperature,
                )
        return decoder_states

    def decode(
            self,
            latent_reprs: Union[torch.Tensor, List[torch.Tensor]],
            max_decode_step: int = 20,
            greedy: bool = True,
            beam_top: int = 1,
            temperature: float = 1.0,
            return_trace: bool = False
    ) -> Union[List[str], List[Tuple[str, List[str]]]]:
        '''
        Input: a batch of vectors for generation.
        Output: the list of smiles of the generated molecules.
        '''
        self.device = next(self.parameters()).device
        latent_reprs = latent_reprs.to(self.device)
        if isinstance(latent_reprs, list):
            latent_reprs = torch.stack(latent_reprs)
        self.batch_size = len(latent_reprs)

        decoder_states: List[DecoderState] = [
            DecoderState(latent_repr=latent_reprs[idx], batch_idx=idx, return_trace=return_trace) for idx in
            range(self.batch_size)]

        decoder_states = self.pick_fisrt_motifs_for_batch(
            latent_reprs=latent_reprs,
            decoder_states=decoder_states,
            greedy=greedy,
            beam_top=beam_top,
            temperature=temperature,
        )

        decoder_states = self.connetion_query_steps_for_batch(
            decoder_states=decoder_states,
            max_decode_step=max_decode_step,
            greedy=greedy,
            beam_top=beam_top,
            temperature=temperature,
        )

        results = [decoder_state.result(return_trace=return_trace) for decoder_state in decoder_states]
        del latent_reprs
        for state in decoder_states:
            del state.latent_repr
            del state.current_graph_data
            del state.motif_node_vecs
            del state.motif_graph_vecs
            del state
        torch.cuda.empty_cache()
        return results

    # 用于外部调试或推理中的单步采样，没有使用于GRPO训练
    def sample(self, state, greedy=False):
        # 当前图状态编码
        graph_data, _ = networkx2data(state.current_graph)
        graph_data = graph_data.to(next(self.parameters()).device)

        # 编码图、获取 query 向量
        query_atom_repr = graph_data.x[state.query_atom]
        graph_repr = self.graph_encoder(graph_data)[1]
        query = self.queryNN(torch.cat((state.latent_repr, graph_repr, query_atom_repr), dim=-1))

        # 🔹Motif部分 logits
        motif_graphs = Batch.from_data_list(self.motif_graphs.to_data_list()).to(self.device)
        node_embed, _ = self.motif_encoder(motif_graphs)
        node_embed = node_embed[self.motif_vocab.get_conns_idx()]
        key_motif = self.keyNN(node_embed)
        motif_logits = torch.matmul(query, key_motif.T)

        # 🔹Cyclize部分 logits
        bond_type = state.query_bond_type
        cyc_cands, cyc_vecs = [], []
        for cyc_cand in state.connections_list[1:]:
            if state.atoms_step_dict[state.query_atom] == state.atoms_step_dict[cyc_cand]:
                continue
            if state.current_graph.nodes[cyc_cand]['dummy_bond_type'] != bond_type:
                continue
            cyc_cands.append(cyc_cand)
            cyc_vecs.append(graph_data.x[cyc_cand])

        if len(cyc_vecs) > 0:
            cyc_vecs = torch.stack(cyc_vecs)
            key_cyc = self.keyNN(cyc_vecs)
            cyc_logits = torch.matmul(query, key_cyc.T)
            logits = torch.cat((motif_logits[self.motif_vocab.bond_type_conns_dict[bond_type]], cyc_logits), dim=-1)
        else:
            logits = motif_logits[self.motif_vocab.bond_type_conns_dict[bond_type]]

        probs = torch.softmax(logits, dim=-1)
        idx = sample_from_distribution(probs, greedy=greedy)
        log_prob = torch.log(probs[idx] + 1e-6).squeeze(0)

        # 🔹确定动作
        if idx < len(self.motif_vocab.bond_type_conns_dict[bond_type]):
            motif_conn_idx = self.motif_vocab.bond_type_conns_dict[bond_type][idx]
            motif_idx, conn_order = self.motif_vocab.from_conn_idx(motif_conn_idx)
            action = ('motif', (motif_idx, conn_order))
        else:
            cyc_idx = idx - len(self.motif_vocab.bond_type_conns_dict[bond_type])
            action = ('cyc', cyc_cands[cyc_idx])

        return action, log_prob
