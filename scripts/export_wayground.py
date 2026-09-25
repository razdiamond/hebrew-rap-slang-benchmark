import json
import random
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "05_gold_benchmark.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "wayground_exports"


def generate_wayground_export(sample_size: int = 30) -> None:
    """Parses the benchmark JSON and exports a random subset to a Wayground-compatible XLSX."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data) > sample_size:
        data = random.sample(data, sample_size)

    rows = []
    for item in data:
        item_id = item.get("id", "")
        artist = item.get("artist", "")
        song = item.get("song_title", "")
        stanza = item.get("context_stanza", "").strip()
        fragment = item.get("fragment", "").strip()

        question_text = (
            f"[{item_id}] 🎵 {artist} - {song}\n\n"
            f"למה התכוון המשורר בשורה: '{fragment}'?\n\n"
            f"הקשר:\n{stanza}"
        )

        options = item.get("options", {})
        label_map = {"A": 1, "B": 2, "C": 3, "D": 4}
        correct_idx = label_map.get(item.get("correct_label"), 1)

        rows.append({
            "Question Text": question_text,
            "Question Type": "Multiple Choice",
            "Option 1": options.get("A", ""),
            "Option 2": options.get("B", ""),
            "Option 3": options.get("C", ""),
            "Option 4": options.get("D", ""),
            "Option 5": "",
            "Correct Answer": correct_idx,
            "Time in seconds": 120,
            "Image Link": "",
            "Answer explanation": ""
        })

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    export_path = OUTPUT_DIR / f"wayground_{sample_size}_random.xlsx"

    df.to_excel(export_path, index=False, sheet_name="Create a Quiz")

    print(
        f"Exported {len(df)} random questions to {export_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    generate_wayground_export()
