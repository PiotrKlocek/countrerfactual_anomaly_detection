from pathlib import Path
import csv

def get_label_from_path(path):
    if "good" in path:
        return 0
    else:
        return 1

def main():
    project_root = Path(__file__).resolve().parents[1]

    csv_path = project_root / "results" / "bottle" / "patchcore" / "results_bottle1.csv"

    correct = 0
    total = 0

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            true_label = get_label_from_path(row["image_path"])
            pred_label = 1 if row["pred_label"] == "True" else 0

            if true_label == pred_label:
                correct += 1

            total += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()