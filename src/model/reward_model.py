from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
import math
import json
from typing import List

# 可选引入 SA score（新的 sascorer 实现）
try:
    from model.sascorer import calculateScore as calc_sa_score
    HAS_SA = True
except ImportError:
    HAS_SA = False


# ---------- 通用函数定义 ----------

def sigmoid(x, center, steepness=1.0):
    """Sigmoid 函数，越接近 center 得分越高"""
    return 1.0 / (1.0 + math.exp(-steepness * (x - center)))

def gaussian(x, mu, sigma):
    """高斯 reward 函数，中心 mu，标准差 sigma"""
    return math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def double_sigmoid(x, left, right, slope=2.0):
    """双 sigmoid 奖励函数：值越接近区间 (left, right) 得分越高"""
    return sigmoid(x, center=left, steepness=slope) * (1.0 - sigmoid(x, center=right, steepness=slope))


def bounded_reward(x, lower, upper):
    """平顶函数：在 [lower, upper] 区间得分为1，否则为0"""
    return 1.0 if lower <= x <= upper else 0.0


# ---------- 主 reward 函数 ----------

def compute_reward(smiles: str, weights: dict) -> float:
    """
    根据输入 SMILES 和指标权重，计算分子 reward。
    支持指标包括：
      - sa:      合成可行性分数，越容易合成越好
      - molwt:   分子量接近 120 越好（适用于)
      - logp:    logP 落在 [0, 4] 越好（亲脂性）
      - tpsa:    TPSA 趋近 60 最佳（极性表面积）
      - valid:   SMILES 是否合法
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return weights.get("valid", 1.0) * 0.0  # 非法结构直接返回0

    reward = 0.0

    for name, weight in weights.items():
        if name == "sa":
            # 合成可行性分数（SA score）
            # 原始范围约为 [1,10]，使用归一化：(10 - sa) / 9
            if not HAS_SA:
                raise ImportError("SA score module not available.")
            sa_score = calc_sa_score(mol)
            score = (10.0 - sa_score) / 9.0
            reward += weight * score

        elif name == "molwt":
            # 分子量（越接近 120 Da 越好）
            molwt = Descriptors.MolWt(mol)
            score = gaussian(molwt, mu=120.0, sigma=30.0)  # 高斯中心=120，宽度=30
            reward += weight * score

        elif name == "logp":
            # logP（亲脂性，越接近 [0,4] 越好）
            logp = Crippen.MolLogP(mol)
            score = double_sigmoid(logp, left=0.0, right=4.0, slope=2.0)
            reward += weight * score

        elif name == "tpsa":
            # 极性表面积 TPSA（越接近 60 越好）
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            score = gaussian(tpsa, mu=60.0, sigma=25.0)
            reward += weight * score

        elif name == "valid":
            # 合法结构给 1 分
            reward += weight * 1.0
        elif name == "cooccur":
            continue

        else:
            raise ValueError(f"Unsupported reward metric: {name}")

    return reward


# ---------- 封装类 ----------

class RewardModel:
    def __init__(self, weights: dict = None, cooccur_path: str = None):
        """
        初始化 reward 评估器。
        支持权重字段：
            sa, molwt, logp, tpsa, valid
        """
        self.weights = weights if weights is not None else {
            "sa": 1.0,
            "molwt": 0.3,
            "logp": 0.2,
            "tpsa": 0.2,
            "valid": 1.0,
            "cooccur": 0.5
        }

        if cooccur_path:
            with open(cooccur_path) as f:
                self.cooccur = json.load(f)
        else:
            self.cooccur = None

    def evaluate(self, smiles_list: list, motif_lists: List[List[str]] = None) -> list:
        """
        对一批 SMILES 分子进行打分
        :return: 每个 SMILES 的 reward 值
        """
        if smiles_list is None or len(smiles_list) == 0:
            return []
        rewards = []
        for i, smi in enumerate(smiles_list):
            r = compute_reward(smi, self.weights)
            if self.cooccur and motif_lists:
                if len(motif_lists) != len(smiles_list):
                    raise ValueError("motif_lists and smiles_list must be of the same length.")
                co_r = self.evaluate_motif_cooccurrence(motif_lists[i])
                r += self.weights.get("cooccur", 0.0) * co_r
            rewards.append(r)
        return rewards

    def evaluate_motif_cooccurrence(self, motif_list: List[str]) -> float:
        if not self.cooccur or len(motif_list) < 2:
            return 0.0

        total_score = 0.0
        count = 0

        for i in range(len(motif_list) - 1):
            m1, m2 = motif_list[i], motif_list[i + 1]
            a, b = sorted([m1, m2])
            freq = self.cooccur.get(a, {}).get(b, 0)
            score = math.log(1 + freq)
            total_score += score
            count += 1

        return total_score / count if count > 0 else 0.0
