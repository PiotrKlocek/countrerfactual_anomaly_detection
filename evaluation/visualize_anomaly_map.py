from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


def unnormalize_image(img_tensor):
    """Convert normalized CHW tensor to HWC numpy image in [0, 1]."""
    img = img_tensor.detach().cpu().numpy()[0]  # C,H,W
    img = np.transpose(img, (1, 2, 0))  # H,W,C

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def main():
    ckpt_path = Path(
        "results/Patchcore/MVTecAD/bottle/v4/weights/lightning/model.ckpt"
    )

    image_path = Path("test/broken_large/000.png")
    output_path = Path("anomaly_map_broken_large_000.png")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Nie znaleziono checkpointu: {ckpt_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")

    dataset = PredictDataset(
        path=str(image_path),
        image_size=(256, 256),
    )

    model = Patchcore()
    engine = Engine(accelerator="auto", devices=1)

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(ckpt_path),
    )

    pred = predictions[0]

    image = unnormalize_image(pred.image)
    anomaly_map = pred.anomaly_map.detach().cpu().numpy()[0]  # H,W

    pred_score = float(pred.pred_score[0])
    pred_label = bool(pred.pred_label[0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(anomaly_map, cmap="jet")
    axes[1].set_title("Anomaly Map")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(image)
    axes[2].imshow(anomaly_map, cmap="jet", alpha=0.45)
    axes[2].set_title(f"Overlay\nscore={pred_score:.4f}, label={pred_label}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Zapisano wizualizację do: {output_path}")


if __name__ == "__main__":
    main()