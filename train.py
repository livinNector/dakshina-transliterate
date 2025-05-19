import argparse
from trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument(
        "--cell_type", type=str, default="LSTM", choices=["RNN", "GRU", "LSTM"]
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_layers", type=int, default=1)

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--lang_code", type=str, default="ta")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_attention", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    train(**dict(args._get_kwargs()))
