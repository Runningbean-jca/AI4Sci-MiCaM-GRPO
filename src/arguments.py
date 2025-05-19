import argparse
from typing import List


# 通过 argparse 提供了一个灵活的方式来控制训练、预处理、模型结构和推理等参数
def parse_arguments():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--preprocess_dir', default='preprocess/')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard/')

    parser.add_argument('--dataset', type=str, default="QM9")
    parser.add_argument('--job_name', type=str, default="")
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--generate_path', type=str, default="samples")

    # hyperparameters
    ## common
    parser.add_argument('--num_workers', type=int, default=60)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)

    ## merging operation learning, motif vocab construction
    parser.add_argument('--num_operations', type=int, default=500)
    parser.add_argument('--num_iters', type=int, default=3000)
    parser.add_argument('--min_frequency', type=int, default=0)
    parser.add_argument('--mp_thd', type=int, default=1e5)

    ## networks
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--atom_embed_size', type=List[int], default=[192, 16, 16, 16, 16])
    parser.add_argument('--edge_embed_size', type=int, default=256)
    parser.add_argument('--motif_embed_size', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--depth', type=int, default=15)
    parser.add_argument('--motif_depth', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--virtual', action='store_true')
    parser.add_argument('--pooling', type=str, default="add")

    ## training
    parser.add_argument('--steps', type=int, default=50000)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_anneal_iter', type=int, default=500)
    parser.add_argument('--lr_anneal_rate', type=float, default=0.99)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)

    parser.add_argument('--beta_warmup', type=int, default=3000)
    parser.add_argument('--beta_min', type=float, default=1e-3)
    parser.add_argument('--beta_max', type=float, default=0.6)
    parser.add_argument('--beta_anneal_period', type=int, default=20000)
    parser.add_argument('--prop_weight', type=float, default=0.5)

    ## grpo training
    # 模型路径
    parser.add_argument("--ckpt_path", type=str, default="output/micam/model.ckpt", help="Path to MiCaM checkpoint")
    parser.add_argument('--num_trajectories_per_z', type=int, default=6, help='Number of trajectories per latent z')
    parser.add_argument("--motif_embed_path", type=str, default="output/micam/motifs_embed.ckpt",
                        help="Path to motif embeddings")
    parser.add_argument("--save_path", type=str, default="output/micam/grpo_best_decoder.ckpt",
                        help="Path to save best decoder")
    parser.add_argument('--batch_size', type=int, default=1)
    # 数据文件路径
    parser.add_argument("--operation_path", type=str, default="data/operations.json")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.txt")
    # 训练参数
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    # 奖励函数
    parser.add_argument('--final_score_weights', type=str,
                        default='{"sa": 1.0, "molwt": 0.3, "logp": 0.2, "tpsa": 0.2, "valid": 1.0, "cooccur": 0.5}',
                        help='JSON string of final reward weights for different properties')
    parser.add_argument("--cooccur_path", type=str, default="motif_cooccur.json",
                        help="Path to motif co-occurrence frequency JSON")
    parser.add_argument('--train_path', type=str, default="data/QM9/train.smiles",
                        help="Path to training SMILES file for benchmarking")

    # inference
    parser.add_argument('--num_sample', type=int, default=10000)

    # generate
    parser.add_argument('--greedy', action='store_true', help='Use greedy decoding mode')


    ## grpo generating
    parser.add_argument("--num_samples", type=int, default=100, help="Number of molecules to generate")
    parser.add_argument("--output", type=str, default="output/generated.grpo.smiles")



    args = parser.parse_args()
    return args
