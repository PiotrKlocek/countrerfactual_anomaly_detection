from pathlib import Path

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Padim


def main():
    project_root = Path(__file__).resolve().parents[2]

    ckpt_path = project_root / "models" / "padim" / "model.ckpt"
    image_path = project_root / "data" / "mvtec" / "bottle" / "test" / "good" / "000.png"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Nie znaleziono checkpointu: {ckpt_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")

    dataset = PredictDataset(
        path=str(image_path),
        image_size=(256, 256),
    )

    model = Padim()
    engine = Engine(accelerator="auto", devices=1)

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(ckpt_path),
    )

    pred = predictions[0]

    print(f"image_path: {pred.image_path}")
    print(f"pred_score: {pred.pred_score}")
    print(f"pred_label: {pred.pred_label}")
    print(f"anomaly_map shape: {pred.anomaly_map.shape}")


if __name__ == "__main__":
    main()