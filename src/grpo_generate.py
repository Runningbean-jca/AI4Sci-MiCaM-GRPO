# grpo_generate.py
# GRPO+MiCaM: 推理脚本，根据训练好的 GRPO 策略模型生成 SMILES 分子

import os
from typing import List

import torch
from datetime import datetime
import argparse
from model.mol_graph import MolGraph
from model.MiCaM_VAE import MiCaM
from model.agent import GRPOAgent
from model.mydataclass import ModelParams
from arguments import parse_arguments

def main():
    args = parse_arguments()
    torch.cuda.set_device(args.cuda)

    print("🚀 Loading vocabulary and operations...")
    MolGraph.load_operations(args.operation_path)
    MolGraph.load_vocab(args.vocab_path)

    print("📦 Loading model...")
    model_params = ModelParams(args)
    model = MiCaM(model_params).cuda()
    model.load_state_dict(torch.load(args.ckpt_path), strict=False)
    model.load_motifs_embed(args.motif_embed_path)
    decoder = model.decoder

    agent = GRPOAgent(decoder)

    print(f"🎯 Start generating {args.num_samples} samples ({'greedy' if args.greedy else 'sampling'})...")
    smiles_list = []
    for _ in range(args.num_samples):
        z = torch.randn(1, args.latent_size).cuda()
        state = decoder.decode(z, greedy=args.greedy)[0]
        smiles_list.append(state)

    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for smi in smiles_list:
            f.write(f"{smi}\n")
    print(f"✅ Saved {len(smiles_list)} molecules to: {args.output}")


if __name__ == "__main__":
    main()