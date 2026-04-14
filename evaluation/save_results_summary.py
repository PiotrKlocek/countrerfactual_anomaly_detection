import csv
from pathlib import Path


def get_label_from_path(path):
    return 0 if "good" in path else 1


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

    results = []

    models = {
        "PatchCore": project_root / "results" / "bottle" / "Patchcore" / "results_bottle.csv",
        "PaDiM": project_root / "results" / "bottle" / "padim" / "results_bottle.csv",
    }

    for model_name, path in models.items():
        if path.exists():
            acc, correct, total = evaluate(path)

            results.append({
                "model": model_name,
                "accuracy": round(acc, 4),
                "correct": correct,
                "total": total
            })
        else:
            print(f"Brak pliku dla {model_name}: {path}")

    # zapis CSV
    output_path = project_root / "results" / "summary.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "correct", "total"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nZapisano podsumowanie: {output_path}")


if __name__ == "__main__":
    main()