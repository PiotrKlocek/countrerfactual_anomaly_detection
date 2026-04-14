import csv
from pathlib import Path


def get_label_from_path(path):
    if "good" in path:
        return 0
    else:
        return 1


def evaluate(csv_path):
    correct = 0
    total = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            true_label = get_label_from_path(row["image_path"])
            pred_label = 1 if row["pred_label"] == "True" else 0

            if true_label == pred_label:
                correct += 1

            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def main():
    project_root = Path(__file__).resolve().parents[1]

    patchcore_csv = project_root / "results" / "bottle" / "Patchcore" / "results_bottle.csv"
    padim_csv = project_root / "results" / "bottle" / "padim" / "results_bottle.csv"

    print("=== EVALUATION ===\n")

    if patchcore_csv.exists():
        acc, correct, total = evaluate(patchcore_csv)
        print(f"PatchCore:")
        print(f"  Accuracy: {acc:.4f} ({correct}/{total})\n")
    else:
        print("PatchCore CSV nie istnieje\n")

    if padim_csv.exists():
        acc, correct, total = evaluate(padim_csv)
        print(f"PaDiM:")
        print(f"  Accuracy: {acc:.4f} ({correct}/{total})\n")
    else:
        print("PaDiM CSV nie istnieje\n")


if __name__ == "__main__":
    main()