from typing import Dict, List, Tuple

import networkx as nx
import rdkit.Chem as Chem
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from rdkit.Chem import RWMol, Atom, BondType


# 将 SMILES 字符串转为 RDKit 的 Mol 对象（即分子结构）
def smiles2mol(smiles: str, sanitize: bool=False) -> Chem.rdchem.Mol:
    if sanitize:
        return Chem.MolFromSmiles(smiles) # 使用 RDKit 自动标准化
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    AllChem.SanitizeMol(mol, sanitizeOps=0)
    return mol

# 将 networkx 图结构（含原子和键信息）转换为 SMILES 字符串
def graph2smiles(fragment_graph: nx.Graph, with_idx: bool=False) -> str:
    motif = Chem.RWMol() # 创建一个可编辑的空分子
    node2idx = {} # 存储每个 graph 节点到 RDKit 原子索引的映射
    for node in fragment_graph.nodes:   
        idx = motif.AddAtom(smarts2atom(fragment_graph.nodes[node]['smarts']))  # 添加节点对应的原子
        if with_idx and fragment_graph.nodes[node]['smarts'] == '*':
            motif.GetAtomWithIdx(idx).SetIsotope(node)  # 设置 isotope 编号，用于拼接标识
        node2idx[node] = idx  # 存储映射
    for node1, node2 in fragment_graph.edges:
        motif.AddBond(node2idx[node1], node2idx[node2], fragment_graph[node1][node2]['bondtype'])  # 添加键
    return Chem.MolToSmiles(motif, allBondsExplicit=True)   # 转为 SMILES，所有键显式表示


# 将 NetworkX 图结构转换为 PyG 的 Data 对象（用于 GNN 输入）
def networkx2data(G: nx.Graph) -> Tuple[Data, Dict[int, int]]:
    num_nodes = G.number_of_nodes()
    mapping = dict(zip(G.nodes(), range(num_nodes))) # 建立新节点编号（0,1,2...）
    
    G = nx.relabel_nodes(G, mapping) # 替换节点编号
    G = G.to_directed() if not nx.is_directed(G) else G   # 转为有向图

    edges = list(G.edges)  # 所有边
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() # PyG 要求的边索引矩阵 [2, E]

    x = torch.tensor([i for _, i in G.nodes(data='label')])   # 每个节点的标签特征
    edge_attr = torch.tensor([[i] for _, _, i in G.edges(data='label')], dtype=torch.long)  # 边属性（键类型）

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)  # 构造 PyG 数据对象

    return data, mapping  # 返回图 + 节点映射关系

# 从一个大分子中提取部分原子，构造片段 SMILES
def fragment2smiles(mol: Chem.rdchem.Mol, indices: List[int]) -> str:
    smiles = Chem.MolFragmentToSmiles(mol, tuple(indices))
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=False))

# 将 SMARTS 字符串转为 RDKit 原子对象
def smarts2atom(smarts: str) -> Chem.rdchem.Atom:
    return Chem.MolFromSmarts(smarts).GetAtomWithIdx(0)

# 将 graph 转为 SMILES，并做可选后处理（修复非法结构）
def mol_graph2smiles(graph: nx.Graph, postprocessing: bool=True) -> str:
    mol = Chem.RWMol()
    graph = nx.convert_node_labels_to_integers(graph)
    node2idx = {}
    for node in graph.nodes:
        idx = mol.AddAtom(smarts2atom(graph.nodes[node]['smarts']))
        node2idx[node] = idx
    for node1, node2 in graph.edges:
        mol.AddBond(node2idx[node1], node2idx[node2], graph[node1][node2]['bondtype'])
    mol = mol.GetMol()
    smiles = Chem.MolToSmiles(mol)
    return postprocess(smiles) if postprocessing else smiles
 
 # 对 SMILES 做 sanitize 尝试，修复非法芳香性、键等错误
def postprocess(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic() and not atom.IsInRing():
                atom.SetIsAromatic(False)   
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                if not (bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic()):
                    bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        
        for _ in range(100):
            problems = Chem.DetectChemistryProblems(mol)
            flag = False
            for problem in problems:
                if problem.GetType() =='KekulizeException':
                    flag = True
                    for atom_idx in problem.GetAtomIndices():
                        mol.GetAtomWithIdx(atom_idx).SetIsAromatic(False)
                    for bond in mol.GetBonds():
                        if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                            if not (bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic()):
                                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol), sanitize=False)
            if flag: continue
            else: break
        
        smi = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        try:
            Chem.SanitizeMol(mol)
        except:
            print(f"{smiles} not valid")
            return "CC"
        smi = Chem.MolToSmiles(mol)
        return smi

# 获取 motif 中的连接点索引（dummy atoms '*') 和排序映射
def get_conn_list(motif: Chem.rdchem.Mol, use_Isotope: bool=False, symm: bool=False) -> Tuple[List[int], Dict[int, int]]:

    ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False, breakTies=False))
    if use_Isotope:
        ordermap = {atom.GetIsotope(): ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*'}
    else:
        ordermap = {atom.GetIdx(): ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*'}
    if len(ordermap) == 0:
        return [], {}
    ordermap = dict(sorted(ordermap.items(), key=lambda x: x[1]))
    if not symm:
        conn_atoms = list(ordermap.keys())
    else:
        cur_order, conn_atoms = -1, []
        for idx, order in ordermap.items():
            if order != cur_order:
                cur_order = order
                conn_atoms.append(idx)
    return conn_atoms, ordermap

# 给 dummy atom 添加唯一编号 isotope（用于拼接对齐）
def label_attachment(smiles: str) -> str:

    mol = Chem.MolFromSmiles(smiles)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    dummy_atoms = [(atom.GetIdx(), ranks[atom.GetIdx()])for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    dummy_atoms.sort(key=lambda x: x[1])
    orders = []
    for (idx, order) in dummy_atoms:
        if order not in orders:
            orders.append(order)
            mol.GetAtomWithIdx(idx).SetIsotope(len(orders))
    return Chem.MolToSmiles(mol)

# 计算 Top-1 / Top-k 准确率
def get_accuracy(scores: torch.Tensor, labels: torch.Tensor):
    _, preds = torch.max(scores, dim=-1)
    acc = torch.eq(preds, labels).float()

    number, indices = torch.topk(scores, k=10, dim=-1)
    topk_acc = torch.eq(indices, labels.view(-1,1)).float()
    return torch.sum(acc) / labels.nelement(), torch.sum(topk_acc) / labels.nelement()

# 从概率分布中采样 motif 索引
def sample_from_distribution(distribution: torch.Tensor, greedy: bool = False, topk: int = 0):
    # print("greedy mode:", greedy)

    if greedy:
        return torch.argmax(distribution, dim=-1)  # shape: [B]

    if topk == 1 or topk == 0:
        return torch.multinomial(distribution, num_samples=1).squeeze(-1)  # shape: [B]

    # Top-k sampling
    topk_values, topk_indices = torch.topk(distribution, topk, dim=-1)
    probs = torch.softmax(topk_values, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return topk_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)



def graph2mol(graph: nx.Graph) -> Chem.Mol:
    """
    将 NetworkX 图结构转换为 RDKit Mol。
    要求节点有 'smarts' 和 'label'，边有 'bondtype'。
    """
    mol = Chem.RWMol()
    node2idx = {}

    try:
        for node in graph.nodes:
            smarts = graph.nodes[node].get('smarts', '[*]')
            atom = Chem.MolFromSmarts(smarts).GetAtomWithIdx(0)
            idx = mol.AddAtom(atom)
            node2idx[node] = idx

        for node1, node2 in graph.edges:
            bondtype = graph[node1][node2].get('bondtype', Chem.rdchem.BondType.SINGLE)
            mol.AddBond(node2idx[node1], node2idx[node2], bondtype)

        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol

    except Exception as e:
        print(f"[ERROR] graph2mol failed: {e}")
        return None