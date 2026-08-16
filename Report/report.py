import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =========================
# Paths
# =========================

GT_PATH = "./GT"
NOISY_PATH = "./NoisyLR"
RESTORED_PATH = "./restored_training_outputs"

OUTPUT_PDF = "./denoising_comparison.pdf"


# =========================
# Helper: NPY -> image
# =========================

def npy_to_image(path):
    arr = np.load(path)
    arr = np.squeeze(arr)

    if arr.ndim == 3 and arr.shape[0] in [1, 3]:
        arr = np.transpose(arr, (1, 2, 0))

    arr = arr.astype(np.float32)
    arr = np.clip(arr, 0, 1)   # no /255 — these are already float, not 8-bit

    return arr


# =========================
# Create PDF
# =========================

gt_files = sorted(
    f for f in os.listdir(GT_PATH)
    if f.endswith(".npy")
)

with PdfPages(OUTPUT_PDF) as pdf:

    for filename in gt_files:

        gt_file = os.path.join(GT_PATH, filename)
        noisy_file = os.path.join(NOISY_PATH, filename)
        restored_file = os.path.join(RESTORED_PATH, filename)

        # Skip if any corresponding file is missing
        if not os.path.exists(noisy_file):
            print(f"Skipping {filename}: NoisyLR not found")
            continue

        if not os.path.exists(restored_file):
            print(f"Skipping {filename}: Restored image not found")
            continue

        # Load images
        gt = npy_to_image(gt_file)
        noisy = npy_to_image(noisy_file)
        restored = npy_to_image(restored_file)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(gt, cmap="gray" if gt.ndim == 2 else None, vmin=0, vmax=1)
        axes[0].set_title("Ground Truth")

        axes[1].imshow(noisy, cmap="gray" if noisy.ndim == 2 else None, vmin=0, vmax=1)
        axes[1].set_title("Noisy LR")

        axes[2].imshow(restored, cmap="gray" if restored.ndim == 2 else None, vmin=0, vmax=1)
        axes[2].set_title("Restored")

        # Remove axes
        for ax in axes:
            ax.axis("off")

        # Filename as page title
        fig.suptitle(filename, fontsize=14)

        plt.tight_layout()

        # Add page to PDF
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        print(f"Added {filename}")


print(f"\nPDF saved to:")
print(OUTPUT_PDF)