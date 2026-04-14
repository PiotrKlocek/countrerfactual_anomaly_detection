from pathlib import Path

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


def main():
    # checkpoint z Twojego ostatniego runa
    ckpt_path = Path(
        "results/Patchcore/MVTecAD/bottle/v4/weights/lightning/model.ckpt"
    )

    # wybierz jeden obraz testowy
    # możesz zmienić tę ścieżkę na dowolny obraz z test/good albo test/<defect_type>
    image_path = Path("test/broken_large/000.png")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Nie znaleziono checkpointu: {ckpt_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")

    # dataset do inferencji na pojedynczym obrazie
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

    print(f"Liczba predykcji: {len(predictions)}")
    print(f"Typ obiektu: {type(predictions[0])}")
    print(predictions[0])

    # spróbuj odczytać najważniejsze pola, jeśli są dostępne
    pred = predictions[0]

    for key in ["pred_score", "pred_label", "anomaly_map", "image_path"]:
        value = getattr(pred, key, None)
        if value is not None:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()