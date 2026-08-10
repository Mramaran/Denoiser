"""Evaluation script placeholder.

This script must be replaced with your evaluation implementation that:
- accepts `--input_dir` and `--output_dir`
- loads your trained model
- runs inference on all images in `--input_dir`
- writes restored images to `--output_dir`

Usage (placeholder):
python evaluate.py --input_dir ./test_images --output_dir ./restored_outputs
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Placeholder evaluation script")
    parser.add_argument("--input_dir", required=True, help="Path to test images directory")
    parser.add_argument("--output_dir", required=True, help="Path to write restored images")
    args = parser.parse_args()

    print("This is a placeholder evaluation script. Replace with your model loading and inference steps.")
    print(f"input_dir={args.input_dir}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
