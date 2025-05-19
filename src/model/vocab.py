import rdkit
import rdkit.Chem as Chem
from typing import List, Tuple, Dict
import torch
from model.utils import smiles2mol, get_conn_list
from collections import defaultdict

# 通用字符串词汇表类
class Vocab(object):
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list # 保存词汇列表（如常见片段 SMILES）
        self.vmap = dict(zip(self.vocab_list, range(len(self.vocab_list))))
        
    def __getitem__(self, smiles):
        return self.vmap[smiles]   # 通过 SMILES 获取其对应的索引

    def get_smiles(self, idx):
        return self.vocab_list[idx]   # 通过索引获取 SMILES

    def size(self):
        return len(self.vocab_list)   # 获取词表大小

# motif 专用词汇表，支持连接点索引与拓扑
class MotifVocab(object):

    def __init__(self, pair_list: List[Tuple[str, str]]):
        self.motif_smiles_list = [motif for _, motif in pair_list] # 提取 motif 的 SMILES 列表
        self.motif_vmap = dict(zip(self.motif_smiles_list, range(len(self.motif_smiles_list))))  # SMILES 到索引的映射

        node_offset, conn_offset, num_atoms_dict, nodes_idx = 0, 0, {}, []
        vocab_conn_dict: Dict[int, Dict[int, int]] = {} # motif_idx -> order -> conn_idx
        conn_dict: Dict[int, Tuple[int, int]] = {}  # conn_idx -> (motif_idx, order)
        bond_type_motifs_dict = defaultdict(list)   # 键类型 -> conn_idx 列表
        for motif_idx, motif_smiles in enumerate(self.motif_smiles_list):
            motif = smiles2mol(motif_smiles)   # 转换为 RDKit 分子对象
            ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False, breakTies=False))   # 原子排序

            cur_orders = []
            vocab_conn_dict[motif_idx] = {}
            for atom in motif.GetAtoms():
                if atom.GetSymbol() == '*' and ranks[atom.GetIdx()] not in cur_orders:
                    bond_type = atom.GetBonds()[0].GetBondType()  # 获取连接键类型（如 SINGLE）
                    vocab_conn_dict[motif_idx][ranks[atom.GetIdx()]] = conn_offset   # 保存连接点标签
                    conn_dict[conn_offset] = (motif_idx, ranks[atom.GetIdx()])  # 保存反查结构
                    cur_orders.append(ranks[atom.GetIdx()])
                    bond_type_motifs_dict[bond_type].append(conn_offset)   # 分 bond 类型组织连接点
                    nodes_idx.append(node_offset)  # 记录该连接点的节点索引
                    conn_offset += 1
                node_offset += 1
            num_atoms_dict[motif_idx] = motif.GetNumAtoms()  # 记录每个 motif 含多少个原子
        self.vocab_conn_dict = vocab_conn_dict  # motif -> order -> conn_idx
        self.conn_dict = conn_dict   # conn_idx -> (motif_idx, order)
        self.nodes_idx = nodes_idx   # 所有连接节点在图中的索引
        self.num_atoms_dict = num_atoms_dict  # motif -> 原子数量
        self.bond_type_conns_dict = bond_type_motifs_dict  # bond_type -> conn_idx 列表


    def __getitem__(self, smiles: str) -> int:
        if smiles not in self.motif_vmap:
            print(f"{smiles} is <UNK>")  # 未知片段提示
        return self.motif_vmap[smiles] if smiles in self.motif_vmap else -1
    
    def get_conn_label(self, motif_idx: int, order_idx: int) -> int:
        return self.vocab_conn_dict[motif_idx][order_idx]  # 给定 motif 和连接点顺序，返回 conn_idx
    
    def get_conns_idx(self) -> List[int]:
        return self.nodes_idx  # 返回所有连接点在图中的索引
    
    def from_conn_idx(self, conn_idx: int) -> Tuple[int, int]:
        return self.conn_dict[conn_idx]  # 通过连接点索引获取对应 motif_idx 和其连接点编号

    def __len__(self):
        return len(self.motif_smiles_list)

# SubMotifVocab 是 motif 的子集词表，用于构建特定任务的子空间搜索
class SubMotifVocab(object):

    def __init__(self, motif_vocab: MotifVocab, sublist: List[int]):
        self.motif_vocab = motif_vocab
        self.sublist = sublist
        self.idx2sublist_map = dict(zip(sublist, range(len(sublist))))

        node_offset, conn_offset, nodes_idx = 0, 0, []
        motif_idx_in_sublist = {}
        vocab_conn_dict: Dict[int, Dict[int, int]] = {}
        for i, mid in enumerate(sublist):
            motif_idx_in_sublist[mid] = i
            vocab_conn_dict[mid] = {}
            for cid in motif_vocab.vocab_conn_dict[mid].keys():
                vocab_conn_dict[mid][cid] = conn_offset
                nodes_idx.append(node_offset + cid)
                conn_offset += 1
            node_offset += motif_vocab.num_atoms_dict[mid]
        self.vocab_conn_dict = vocab_conn_dict
        self.nodes_idx = nodes_idx
        self.motif_idx_in_sublist_map = motif_idx_in_sublist
    
    def motif_idx_in_sublist(self, motif_idx: int):
        return self.motif_idx_in_sublist_map[motif_idx]  # 查询某 motif 在子集内的编号

    def get_conn_label(self, motif_idx: int, order_idx: int):
        return self.vocab_conn_dict[motif_idx][order_idx]  # 子集内连接点标签
    
    def get_conns_idx(self):
        return self.nodes_idx  # 子集中所有连接点节点索引

    



