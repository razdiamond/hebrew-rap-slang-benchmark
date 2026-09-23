"""
Central configuration for all benchmarked models, endpoints, and credentials.

Execution note:
Inference and evaluation pipelines are fully cached and idempotent. Existing models
with completed cache files in `data/` are automatically skipped, so adding a new
entry only runs inference for the newly added model.
"""
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# --- Credentials ---
# Local Ollama instances require a non-empty dummy token for client compatibility
OLLAMA_KEY = "ollama"
GEMINI_KEY = os.environ.get(key="GEMINI_API_KEY", default="offline_dummy_key")
OPENROUTER_KEY = os.environ.get(key="OPENROUTER_KEY", default="offline_dummy_key")

# --- Provider Endpoints ---
OLLAMA_BASE_URL = "http://localhost:11434/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS_CONFIG = [
    {
        "real_name": "aminadaven/dictalm2.0-instruct:Q4_K_M",
        "clean_name": "dictalm_2.0_instruct_q4",
        "creator": "dicta-il",
        "params_b": 7.0,
        "release_date": "2024-07",
        "type": "Open",
        "base_url": OLLAMA_BASE_URL,
        "api_key": OLLAMA_KEY,
    },
    {
        "real_name": "DictaLM-3.0-1.7B-Instruct:latest",
        "clean_name": "dictalm_3.0_1.7b_instruct",
        "creator": "dicta-il",
        "params_b": 1.7,
        "release_date": "2025-02",
        "type": "Open",
        "base_url": OLLAMA_BASE_URL,
        "api_key": OLLAMA_KEY,
    },
    {
        "real_name": "hf.co/dicta-il/DictaLM-3.0-Nemotron-12B-Instruct-GGUF:Q4_K_M",
        "clean_name": "dictalm_3.0_nemotron_12b_instruct_q4",
        "creator": "dicta-il",
        "params_b": 12.0,
        "release_date": "2025-02",
        "type": "Open",
        "base_url": OLLAMA_BASE_URL,
        "api_key": OLLAMA_KEY,
    },
    {
        "real_name": "gemma3:4b",
        "clean_name": "gemma_3_4b_it",
        "creator": "google",
        "params_b": 4.0,
        "release_date": "2025-02",
        "type": "Open",
        "base_url": OLLAMA_BASE_URL,
        "api_key": OLLAMA_KEY,
    },
    {
        "real_name": "qwen3:8b",
        "clean_name": "qwen3_8b_instruct",
        "creator": "qwen",
        "params_b": 8.0,
        "release_date": "2025-04",
        "type": "Open",
        "base_url": OLLAMA_BASE_URL,
        "api_key": OLLAMA_KEY,
    },
    {
        "real_name": "llama3.2:latest",
        "clean_name": "llama_3.2_3b_instruct",
        "creator": "meta-llama",
        "params_b": 3.0,
        "release_date": "2024-09",
        "type": "Open",
        "base_url": OLLAMA_BASE_URL,
        "api_key": OLLAMA_KEY,
    },
    {
        "real_name": "hebatron:30b-q4",
        "clean_name": "hebatron_30b_q4",
        "creator": "hebarabnlpproject",
        "params_b": 30.0,
        "release_date": "2026-05",
        "type": "Open",
        "base_url": OLLAMA_BASE_URL,
        "api_key": OLLAMA_KEY,
    },
    {
        "real_name": "gemini-3.8-flash",
        "clean_name": "gemini_3.8_flash",
        "creator": "google",
        "params_b": None,
        "release_date": "2026-08",
        "type": "Proprietary",
        "base_url": GEMINI_BASE_URL,
        "api_key": GEMINI_KEY,
    },
    {
        "real_name": "gemini-3.1-pro-preview",
        "clean_name": "gemini_3.1_pro_preview",
        "creator": "google",
        "params_b": None,
        "release_date": "2026-04",
        "type": "Proprietary",
        "base_url": GEMINI_BASE_URL,
        "api_key": GEMINI_KEY,
    },
    {
        "real_name": "gemma-4-31b-it",
        "clean_name": "gemma_4_31b_it",
        "creator": "google",
        "params_b": 31.0,
        "release_date": "2026-05",
        "type": "Open",
        "base_url": GEMINI_BASE_URL,
        "api_key": GEMINI_KEY,
    },
    {
        "real_name": "gemini-3-flash-preview",
        "clean_name": "gemini_3_flash_preview",
        "creator": "google",
        "params_b": None,
        "release_date": "2025-06",
        "type": "Proprietary",
        "base_url": GEMINI_BASE_URL,
        "api_key": GEMINI_KEY,
    },
    {
        "real_name": "gemini-3.5-flash-lite",
        "clean_name": "gemini_3.5_flash_lite",
        "creator": "google",
        "params_b": None,
        "release_date": "2026-02",
        "type": "Proprietary",
        "base_url": GEMINI_BASE_URL,
        "api_key": GEMINI_KEY,
    },
    {
        "real_name": "gemma-4-26b-a4b-it",
        "clean_name": "gemma_4_26b_a4b_it",
        "creator": "google",
        "params_b": 26.0,
        "release_date": "2026-05",
        "type": "Open",
        "base_url": GEMINI_BASE_URL,
        "api_key": GEMINI_KEY,
    },
    {
        "real_name": "openai/gpt-5.6-luna",
        "clean_name": "gpt_5.6_luna",
        "creator": "openai",
        "params_b": None,
        "release_date": "2026-07",
        "type": "Proprietary",
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "openai/gpt-5",
        "clean_name": "gpt_5",
        "creator": "openai",
        "params_b": None,
        "release_date": "2025-09",
        "type": "Proprietary",
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "mistralai/mistral-large-2407",
        "clean_name": "mistral_large_2407",
        "creator": "mistralai",
        "params_b": 123.0,
        "release_date": "2024-07",
        "type": "Open",
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "meta-llama/llama-3.3-70b-instruct",
        "clean_name": "llama_3.3_70b_instruct",
        "creator": "meta-llama",
        "params_b": 70.0,
        "release_date": "2024-12",
        "type": "Open",
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "qwen/qwen-2.5-72b-instruct",
        "clean_name": "qwen_2.5_72b_instruct",
        "creator": "qwen",
        "params_b": 72.0,
        "release_date": "2024-09",
        "type": "Open",
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "cohere/command-a",
        "clean_name": "command_a",
        "creator": "cohere",
        "params_b": None,
        "release_date": "2025-04",
        "type": "Proprietary",
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "deepseek/deepseek-v4-flash",
        "clean_name": "deepseek_v4_flash",
        "creator": "deepseek",
        "params_b": None,
        "release_date": "2026-04",
        "type": "Open",
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "deepseek/deepseek-v4.1-flash",
        "clean_name": "deepseek_v4.1_flash",
        "creator": "deepseek",
        "params_b": None,
        "release_date": "2026-09",
        "type": "Open",
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "xiaomi/mimo-v2.5",
        "clean_name": "mimo_v2.5",
        "creator": "xiaomi",
        "params_b": None,
        "release_date": "2026-03",
        "type": "Proprietary",
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "meta/muse-spark-1.3-contributor",
        "clean_name": "muse_spark_1.3_contributor",
        "creator": "meta",
        "params_b": None,
        "release_date": "2026-07",
        "type": "Proprietary",
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "cohere/command-r-08-2024",
        "clean_name": "command_r_08_2024",
        "creator": "cohere",
        "params_b": 35.0,
        "release_date": "2024-08",
        "type": "Open",
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "qwen/qwen-2.5-coder-32b-instruct",
        "clean_name": "qwen_2.5_coder_32b_instruct",
        "creator": "qwen",
        "params_b": 32.5,
        "release_date": "2024-11",
        "type": "Open",
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_KEY,
    },
]

READY_MODELS: set[str] = {m["clean_name"] for m in MODELS_CONFIG}
MODEL_METADATA: dict[str, dict[str, Any]] = {m["clean_name"]: m for m in MODELS_CONFIG}
