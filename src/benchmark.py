import random
import torch
from fcd_torch import FCD
from rdkit import RDLogger
import json

from arguments import parse_arguments
from model.mydataclass import ModelParams, Paths

# 可选：关闭 RDKit 的警告信息
RDLogger.DisableLog('rdApp.*')

def load_smiles_from_file(filepath):
    with open(filepath, 'r') as f:
        smiles = [line.strip() for line in f if line.strip()]
    return smiles

if __name__ == '__main__':
    args = parse_arguments()
    paths = Paths(args)
    model_params = ModelParams(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.set_device(args.cuda)

    # 载入生成的 SMILES（不重新生成）
    generated_smiles = load_smiles_from_file(paths.generate_path)

    # 载入真实分子
    reference_smiles = load_smiles_from_file(paths.train_path)

    # 使用 fcd_torch 计算 FCD
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fcd_metric = FCD(device=device)
    score = fcd_metric(generated_smiles, reference_smiles)

    print(f"\n🧪 FCD Score = {score:.4f}")

    # 保存结果到 JSON
    with open(paths.benchmark_path, 'w') as out_file:
        json.dump({"FCD": score}, out_file, indent=2)