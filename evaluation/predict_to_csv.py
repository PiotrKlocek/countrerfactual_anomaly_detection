from pathlib import Path
import csv

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


def main():
    project_root = Path(__file__).resolve().parent.parent

    ckpt_path = project_root / "models" / "patchcore" / "model.ckpt"
    data_path = project_root / "data" / "mvtec" / "bottle" / "test"
    output_path = project_root / "results" / "bottle" / "patchcore" / "results_bottle1.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Nie znaleziono checkpointu: {ckpt_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Nie znaleziono danych: {data_path}")

    dataset = PredictDataset(
        path=str(data_path),
        image_size=(256, 256),
    )

    model = Patchcore()
    engine = Engine(accelerator="auto", devices=1)

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(ckpt_path),
    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "pred_score", "pred_label"])

        for pred in predictions:
            image_path = pred.image_path[0]
            score = float(pred.pred_score[0])
            label = bool(pred.pred_label[0])
            writer.writerow([image_path, score, label])

    print(f"Zapisano: {output_path}")


if __name__ == "__main__":
    main()