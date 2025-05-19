# from rdkit import Chem
# from rdkit.Chem import Descriptors
# import pandas as pd
#
# # 读取原始 zinc250k 数据集
# df = pd.read_csv("zinc250k.csv")  # 请确保该文件路径正确
# smiles_list = df["smiles"].tolist()
#
# # 清洗函数：合法 + 去盐 + 分子量筛选 + 去手性
# def clean_smiles(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if not mol:
#         return None
#     mw = Descriptors.MolWt(mol)
#     if mw < 100 or mw > 600:
#         return None
#     if '.' in smiles:
#         return None
#     # 转为 canonical SMILES，不带手性信息
#     return Chem.MolToSmiles(mol, isomericSmiles=False)
#
# # 执行清洗
# cleaned = list(filter(None, map(clean_smiles, smiles_list)))
#
# # 保存为 zinc.smiles 文件（每行一个 SMILES）
# with open("zinc.smiles", "w") as f:
#     for s in cleaned:
#         f.write(s + "\n")
#
# print(f"✅ 清洗完成，zinc.smiles 中共有 {len(cleaned)} 条有效分子。")



import random

# 读取清洗后的 zinc.smiles
with open("zinc.smiles", "r") as f:
    smiles_all = [line.strip() for line in f if line.strip()]

# 随机打乱
random.shuffle(smiles_all)

# 按 9:1 划分
split_index = int(len(smiles_all) * 0.9)
train_smiles = smiles_all[:split_index]
valid_smiles = smiles_all[split_index:]

# 保存到新文件
with open("train.smiles", "w") as f:
    for s in train_smiles:
        f.write(s + "\n")

with open("valid.smiles", "w") as f:
    for s in valid_smiles:
        f.write(s + "\n")

print(f"✅ 划分完成：训练集 {len(train_smiles)} 条，验证集 {len(valid_smiles)} 条")