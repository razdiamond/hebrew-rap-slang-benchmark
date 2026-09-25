import json
import time
from pathlib import Path
from typing import Callable, Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from tqdm.auto import tqdm

from models_config import MODELS_CONFIG

# --- Paths ---
BENCHMARK_PATH = Path("../data/processed/05_gold_benchmark.json")

RESULTS_DIR = Path("../data/results/multichoice")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OPEN_RESULTS_DIR = Path("../data/results/open_ended")
OPEN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -- Prompts --
MC_EVAL_SYSTEM_PROMPT = """אתה מומחה לניתוח טקסט, תרבות וסלנג ישראלי.
לפניך קטע משיר היפ הופ ישראלי, שורה ספציפית מתוכו, ו-4 אפשרויות הסבר (A, B, C, D).
עליך לבחור את האפשרות המתאימה ביותר לפי ההקשר.

השב במבנה JSON תקין הכולל מפתח אחד בלבד:
{
  "selected_option": "אות התשובה הנבחרת בלבד (A, B, C או D)"
}"""

OPEN_EVAL_SYSTEM_PROMPT = """אתה עוזר וירטואלי מומחה למוזיקה, היפ-הופ וסלנג ישראלי.
עליך להסביר בצורה מדויקת, תמציתית וברורה את המשמעות של השורה המבוקשת מתוך השיר.
ענה בעברית בלבד ובלי הקדמות מיותרות."""


# --- Formatting & Extraction Callbacks ---
def format_mc_user_prompt(item: dict) -> str:
    options_text = "\n".join([f"{k}. {v}" for k, v in item["options"].items()])
    return f"""אמן: {item['artist']}
שיר: {item['song_title']}

הקשר מתוך השיר:
{item['context_stanza']}

השורה לניתוח:
"{item['fragment']}"

אפשרויות:
{options_text}

איזו אפשרות היא הנכונה ביותר? השב ב-JSON בלבד."""


def extract_mc_record(item: dict, raw_output: str) -> dict[str, Any]:
    return {
        "id": item["id"],
        "fragment": item["fragment"],
        "correct_label": item["correct_label"],
        "raw_model_response": raw_output,
    }


def format_open_user_prompt(item: dict) -> str:
    return f"""אמן: {item['artist']}
שיר: {item['song_title']}

הקשר מתוך השיר:
{item['context_stanza']}

השורה לניתוח:
"{item['fragment']}"

הסבר במדויק ובתמציתיות למה התכוון האמן בשורה זו:"""


def extract_open_record(item: dict, raw_output: str) -> dict[str, Any]:
    return {
        "id": item["id"],
        "fragment": item["fragment"],
        "ground_truth": item["raw_genius_explanation"],
        "raw_model_response": raw_output,
    }


# --- Core Inference Engine ---
def process_inference_task(
        config: dict,
        benchmark_data: list[dict],
        results_dir: Path,
        task_label: str,
        system_prompt: str,
        prompt_formatter: Callable[[dict], str],
        require_json: bool,
        record_extractor: Callable[[dict, str], dict],
) -> str:
    """Generic inference engine handling API requests, retries, and atomic caching."""
    model_name = config["real_name"]
    clean_name = config["clean_name"]
    output_file = results_dir / f"{clean_name}.json"

    client = OpenAI(
        base_url=config["base_url"],
        api_key=config.get("api_key", "dummy-key"),
        max_retries=0,
        timeout=500.0,
    )

    evaluated_records = []

    # Load existing valid records (ignoring previous crashes to force retries)
    if output_file.exists():
        with output_file.open(mode="r", encoding="utf-8") as f:
            for r in json.load(f):
                if not r.get("raw_model_response", "").startswith("Exception:"):
                    evaluated_records.append(r)

    completed_ids = {r["id"] for r in evaluated_records}
    pending_items = [item for item in benchmark_data if item["id"] not in completed_ids]

    print(f"\n--- {task_label} Inference: {clean_name} ({model_name}) ---")
    print(
        f"Total: {len(benchmark_data)} | "
        f"Done: {len(completed_ids)} | "
        f"Pending: {len(pending_items)}"
    )

    if not pending_items:
        return f"{len(benchmark_data)}/{len(benchmark_data)} Done"

    for item in tqdm(
            pending_items,
            desc=f"{task_label}: {clean_name}",
            mininterval=2.0,
    ):
        user_prompt = prompt_formatter(item)
        raw_output = ""
        success = False

        for attempt in range(2):
            try:
                messages: list[ChatCompletionMessageParam] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                if require_json:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=4000,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                else:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=4000,
                    )

                content = response.choices[0].message.content
                raw_output = content.strip() if content else ""
                success = True
                break
            except Exception as e:
                raw_output = f"Exception: {str(e)}"
                time.sleep(2)

        if not success:
            print(
                f"\nWarning: Model API crashed on item {item['id']}. "
                "Logging as Exception."
            )

        # Extract record and cache atomically
        record = record_extractor(item, raw_output)
        evaluated_records.append(record)

        with output_file.open(mode="w", encoding="utf-8") as f:
            json.dump(evaluated_records, f, ensure_ascii=False, indent=2)

        # Rate limiting for cloud models (skip for localhost)
        base_url = config.get("base_url", "")
        is_local = any(prefix in base_url for prefix in ["localhost", "127.0.0.1"])

        if not is_local:
            time.sleep(4.0)

    total_valid = sum(
        1
        for r in evaluated_records
        if not r.get("raw_model_response", "").startswith("Exception:")
    )
    return f"{total_valid}/{len(benchmark_data)} Done"


# --- Main Execution Flow ---
def main():
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Cannot find benchmark file at: {BENCHMARK_PATH}")

    with BENCHMARK_PATH.open(mode="r", encoding="utf-8") as f:
        benchmark_items = json.load(f)

    summary = {}

    print("Starting Evaluation Pipeline...")

    for config in MODELS_CONFIG:
        clean_name = config["clean_name"]
        summary[clean_name] = {"mc": "ERROR", "open": "ERROR"}

        # 1. Multiple-Choice Phase
        try:
            mc_status = process_inference_task(
                config=config,
                benchmark_data=benchmark_items,
                results_dir=RESULTS_DIR,
                task_label="Multiple-Choice",
                system_prompt=MC_EVAL_SYSTEM_PROMPT,
                prompt_formatter=format_mc_user_prompt,
                require_json=True,
                record_extractor=extract_mc_record,
            )
            summary[clean_name]["mc"] = mc_status
        except Exception as e:
            print(f"Multiple-choice pipeline error for {clean_name}: {e}")

        # 2. Open-Ended Phase
        try:
            open_status = process_inference_task(
                config=config,
                benchmark_data=benchmark_items,
                results_dir=OPEN_RESULTS_DIR,
                task_label="Open-Ended",
                system_prompt=OPEN_EVAL_SYSTEM_PROMPT,
                prompt_formatter=format_open_user_prompt,
                require_json=False,
                record_extractor=extract_open_record,
            )
            summary[clean_name]["open"] = open_status
        except Exception as e:
            print(f"Open-ended pipeline error for {clean_name}: {e}")

    # Final Combined Status Report
    print("\n" + "=" * 65)
    print(f"{'MODEL':<30} | {'MULTIPLE-CHOICE':<15} | {'OPEN-ENDED':<15}")
    print("=" * 65)
    for model_name, stats in summary.items():
        print(f"{model_name:<30} | {stats['mc']:<15} | {stats['open']:<15}")
    print("=" * 65)


if __name__ == "__main__":
    main()
