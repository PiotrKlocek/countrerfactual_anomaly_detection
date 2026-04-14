from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


def main():
    datamodule = MVTecAD(
        root="..",              # bo skrypt uruchamiasz z folderu bottle
        category="bottle",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
    )

    model = Patchcore()

    engine = Engine(
        max_epochs=1,
        accelerator="auto",
        devices=1,
    )

    engine.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()