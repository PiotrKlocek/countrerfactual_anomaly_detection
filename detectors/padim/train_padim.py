from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim


def main():
    project_root = Path(__file__).resolve().parents[2]

    data_root = project_root / "data" / "mvtec"
    results_root = project_root / "results" / "bottle" / "padim"

    model = Padim()

    datamodule = MVTecAD(
        root=str(data_root),
        category="bottle",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
    )

    engine = Engine(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        default_root_dir=str(results_root),
    )

    engine.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()