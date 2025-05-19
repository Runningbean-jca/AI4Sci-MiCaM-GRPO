# MiCaM 分子生成模型的推理脚本
import random
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from arguments import parse_arguments
from model.MiCaM_VAE import MiCaM
from model.mydataclass import ModelParams, Paths

if __name__ == '__main__':
    
    args = parse_arguments()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    paths = Paths(args)
    model_params = ModelParams(args)
    tb = SummaryWriter(log_dir=paths.tensorboard_dir)
    
    generator = MiCaM.load_generator(model_params, paths)
    print(f"[{datetime.now()}] Begin generating...")
    samples = generator.generate(args.num_sample, greedy=args.greedy)
    print(f"[{datetime.now()}] End generating...")

    mode_tag = "greedy" if args.greedy else "sampled"
    output_path = f"{paths.generate_path}.{mode_tag}.smiles"

    with open(output_path, "w") as f:
        for smi in samples:
            f.write(f"{smi}\n")
    
    print(f"Saved {len(samples)} samples to {output_path}")

