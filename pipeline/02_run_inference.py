import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

BENCHMARK_PATH = Path("../data/processed/gold_benchmark_150.json")
RESULTS_DIR = Path("../data/results/multichoice")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OPEN_RESULTS_DIR = Path("../data/results/open_ended")
OPEN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_CONFIG = [
    {
        "real_name": "aminadaven/dictalm2.0-instruct:Q4_K_M",
        "clean_name": "DictaLM_2.0_Instruct_Q4",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama"
    },
    {
        "real_name": "DictaLM-3.0-1.7B-Instruct:latest",
        "clean_name": "DictaLM_3.0_1.7B_Instruct",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama"
    },
    {
        "real_name": "hf.co/dicta-il/DictaLM-3.0-Nemotron-12B-Instruct-GGUF:Q4_K_M",
        "clean_name": "DictaLM_3.0_Nemotron_12B_Q4",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama"
    },
    {
        "real_name": "gemma3:4b",
        "clean_name": "Gemma_3_4B",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama"
    },
    {
        "real_name": "qwen3:8b",
        "clean_name": "Qwen_3_8B",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama"
    },
    {
        "real_name": "llama3.2:latest",
        "clean_name": "Llama_3_2",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama"
    },
    # --- ADDED: The 3 new Gemini API models ---
    {
        "real_name": "gemini-3.8-flash",
        "clean_name": "Gemini_3.8_Flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_KEY
    },
    {
        "real_name": "gemini-3.1-pro-preview",
        "clean_name": "Gemini_3.1_Pro",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_KEY
    },
    {
        "real_name": "gemma-4-31b-it",
        "clean_name": "Gemma_4_31B",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_KEY
    },
    # {
    #     "real_name": "hebatron:30b-q4",
    #     "clean_name": "Hebatron_30B",
    #     "base_url": "http://localhost:11434/v1",
    #     "api_key": "ollama"
    # },
]

local_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    max_retries=0,
    timeout=500.0
)

MC_EVAL_SYSTEM_PROMPT = """אתה מומחה לניתוח טקסט, תרבות וסלנג ישראלי.
לפניך קטע משיר היפ הופ ישראלי, שורה ספציפית מתוכו, ו-4 אפשרויות הסבר (A, B, C, D).
עליך לבחור את האפשרות המתאימה ביותר לפי ההקשר.

השב במבנה JSON תקין הכולל מפתח אחד בלבד:
{
  "selected_option": "אות התשובה הנבחרת בלבד (A, B, C או D)"
}"""

def format_eval_user_prompt(item: dict) -> str:
    options_text = "\n".join([f"{k}. {v}" for k, v in item["options"].items()])
    return f"""אמן: {item['artist']}
שיר: {item['song_title']}

הקשר מתוך השיר:
{item.get('context_stanza', '')}

השורה לניתוח:
"{item['fragment']}"

אפשרויות:
{options_text}

איזו אפשרות היא הנכונה ביותר? השב ב-JSON בלבד."""

def run_inference_pipeline(
    config: dict,
    benchmark_data: list[dict],
    results_dir: Path
):
    model_name = config["real_name"]
    clean_name = config["clean_name"]
    output_file = results_dir / f"{clean_name}.json"

    client = OpenAI(
        base_url=config["base_url"],
        api_key=config.get("api_key", "dummy-key"),
        max_retries=0,
        timeout=500.0
    )

    evaluated_records = []

    # Load existing records, but filter out timeouts and API crashes
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            raw_records = json.load(f)
            for r in raw_records:
                response = r.get("raw_model_response", "")
                # If it's a real API crash/timeout, DO NOT add it to evaluated_records.
                # This ensures it gets put back into pending_items.
                if response.startswith("Exception:"):
                    continue
                evaluated_records.append(r)

    completed_ids = {r["id"] for r in evaluated_records}
    pending_items = [item for item in benchmark_data if item["id"] not in completed_ids]

    print(f"\n{'='*55}")
    print(f"Inference: {clean_name} ({model_name})")
    print(f"Total: {len(benchmark_data)} | Done: {len(completed_ids)} | Pending: {len(pending_items)}")
    print(f"{'='*55}")

    for item in tqdm(pending_items, desc=f"Infer: {clean_name}", mininterval=2.0):
        user_prompt = format_eval_user_prompt(item)
        raw_output = ""
        success = False

        # Attempt the API call (retrying once if it drops connection)
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": MC_EVAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                raw_output = response.choices[0].message.content.strip()
                success = True
                break  # Exit the retry loop on success
            except Exception as e:
                raw_output = f"Exception: {str(e)}"
                time.sleep(2)

        if not success:
            print(f"\n[!] Model API Crashed on {item['id']}. Logging as Exception.")
            print(f"    Error snippet: {raw_output[:100]}...")

        # Record only the raw inference data
        record = {
            "id": item["id"],
            "fragment": item["fragment"],
            "correct_label": item["correct_label"],
            "raw_model_response": raw_output
        }

        evaluated_records.append(record)

        # Save atomically
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(evaluated_records, f, ensure_ascii=False, indent=2)
        if "googleapis" in config["base_url"]:
            time.sleep(4.0)

    total_target = len(benchmark_data)
    total_valid = len([r for r in evaluated_records if not r.get("raw_model_response", "").startswith("Exception:")])

    print(f"\n[✓] Finished {clean_name}!")
    print(f"    Valid Inferences: {total_valid}/{total_target}")
    return f"{total_valid}/{total_target} Done"

if not BENCHMARK_PATH.exists():
    raise FileNotFoundError(f"Cannot find benchmark file at: {BENCHMARK_PATH}")

with BENCHMARK_PATH.open("r", encoding="utf-8") as f:
    benchmark_items = json.load(f)

summary_results = {}

for config in MODELS_CONFIG:
    try:
        status = run_inference_pipeline(
            config=config,
            benchmark_data=benchmark_items,
            results_dir=RESULTS_DIR
        )
        summary_results[config["clean_name"]] = status
    except Exception as e:
        print(f"Pipeline error for {config['clean_name']}: {e}")
        summary_results[config["clean_name"]] = "CRASHED"

print("\n" + "="*45)
print("INFERENCE PIPELINE STATUS")
print("="*45)
for clean_name, status in summary_results.items():
    print(f"{clean_name:<30} | {status}")
print("="*45)

OPEN_EVAL_SYSTEM_PROMPT = """אתה עוזר וירטואלי מומחה למוזיקה, היפ-הופ וסלנג ישראלי.
עליך להסביר בצורה מדויקת, תמציתית וברורה את המשמעות של השורה המבוקשת מתוך השיר.
ענה בעברית בלבד ובלי הקדמות מיותרות."""

def format_open_eval_user_prompt(item: dict) -> str:
    context_stanza = item.get("stanzas", item.get("stanza", item.get("lyrics", item.get("context", ""))))
    if isinstance(context_stanza, list):
        context_stanza = "\n".join(context_stanza)

    return f"""אמן: {item.get('artist', 'לא ידוע')}
שיר: {item.get('song_title', 'לא ידוע')}

הקשר מתוך השיר:
{context_stanza}

השורה לניתוח:
"{item.get('fragment', '')}"

הסבר במדויק ובתמציתיות למה התכוון האמן בשורה זו:"""

def run_open_inference_pipeline(
    config: dict,
    benchmark_data: list[dict],
    results_dir: Path
):
    model_name = config["real_name"]
    clean_name = config["clean_name"]
    output_file = results_dir / f"{clean_name}.json"

    # --- ADDED: Dynamic client instantiation exactly like run_inference_pipeline ---
    client = OpenAI(
        base_url=config["base_url"],
        api_key=config.get("api_key", "dummy-key"),
        max_retries=0,
        timeout=500.0
    )

    evaluated_records = []

    # Load existing records, filtering out crashes so they get retried
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            raw_records = json.load(f)
            for r in raw_records:
                if r.get("raw_model_response", "").startswith("Exception:"):
                    continue
                evaluated_records.append(r)

    completed_ids = {r["id"] for r in evaluated_records}
    pending_items = [item for item in benchmark_data if item["id"] not in completed_ids]

    print(f"\n{'='*55}")
    print(f"Open-Ended Inference: {clean_name} ({model_name})")
    print(f"Total: {len(benchmark_data)} | Done: {len(completed_ids)} | Pending: {len(pending_items)}")
    print(f"{'='*55}")

    if not pending_items:
        return f"{len(benchmark_data)}/{len(benchmark_data)} Done"

    for item in tqdm(pending_items, desc=f"Open Infer: {clean_name}", mininterval=2.0):
        user_prompt = format_open_eval_user_prompt(item)
        raw_output = ""
        success = False

        # Attempt the API call (Notice: response_format is NOT constrained to json_object)
        for attempt in range(2):
            try:
                # --- CHANGED: Now using 'client' instead of 'local_client' ---
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": OPEN_EVAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                )
                raw_output = response.choices[0].message.content.strip()
                success = True
                break
            except Exception as e:
                raw_output = f"Exception: {str(e)}"
                time.sleep(2)

        if not success:
            print(f"\n[!] Model API Crashed on {item.get('id')}. Logging as Exception.")

        # Capture the ground truth correct explanation so we have it for Notebook 2
        raw_genius_explanation = item.get("raw_genius_explanation", "")

        record = {
            "id": item.get("id"),
            "fragment": item.get("fragment"),
            "ground_truth": raw_genius_explanation,
            "raw_model_response": raw_output
        }

        evaluated_records.append(record)

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(evaluated_records, f, ensure_ascii=False, indent=2)
        if "googleapis" in config["base_url"]:
            time.sleep(4.0)

    total_target = len(benchmark_data)
    total_valid = len([r for r in evaluated_records if not r.get("raw_model_response", "").startswith("Exception:")])

    print(f"\n[✓] Finished Open-Ended for {clean_name}!")
    return f"{total_valid}/{total_target} Done"

print("\nStarting Open-Ended (Free Response) Inference Phase...")
open_summary = {}

for config in MODELS_CONFIG:
    try:
        status = run_open_inference_pipeline(
            config=config,
            benchmark_data=benchmark_items,
            results_dir=OPEN_RESULTS_DIR
        )
        open_summary[config["clean_name"]] = status
    except Exception as e:
        print(f"Pipeline error for {config['clean_name']}: {e}")
        open_summary[config["clean_name"]] = "CRASHED"

print("\n" + "="*45)
print("OPEN-ENDED PIPELINE STATUS")
print("="*45)
for clean_name, status in open_summary.items():
    print(f"{clean_name:<30} | {status}")
print("="*45)