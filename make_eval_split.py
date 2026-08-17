"""Freeze a held-out evaluation split, once.

Report/ is byte-identical to the first 101 training images, so it cannot be
used to score a trained model. This writes a seeded 10% holdout that training
must never see.

Usage:
    python make_eval_split.py --gt_dir ../Dataset/train/train/GT
"""

import argparse
import os
import random


def main():
    parser = argparse.ArgumentParser(description="Freeze a held-out eval split")
    parser.add_argument("--gt_dir", default="../Dataset/train/train/GT",
                        help="Directory of ground-truth .npy files")
    parser.add_argument("--out", default="eval_split.txt",
                        help="Where to write the held-out filename list")
    parser.add_argument("--fraction", type=float, default=0.1,
                        help="Fraction held out (default: 0.1)")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.gt_dir) if f.endswith(".npy"))
    if not files:
        raise RuntimeError(f"No .npy files found in {args.gt_dir}")

    rng = random.Random(args.seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(len(files) * args.fraction))
    held_out = sorted(shuffled[:n_val])

    with open(args.out, "w") as f:
        f.write("\n".join(held_out) + "\n")

    print(f"[EvalSplit] {len(files)} total, {len(held_out)} held out -> {args.out}")


if __name__ == "__main__":
    main()
