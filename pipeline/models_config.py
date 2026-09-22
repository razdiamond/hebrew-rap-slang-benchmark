"""
Central configuration for all benchmarked models, endpoints, and credentials.
"""
import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_KEY = os.environ.get(key="GEMINI_API_KEY", default="offline_dummy_key")
OPENROUTER_KEY = os.environ.get(key="OPENROUTER_KEY", default="offline_dummy_key")

MODELS_CONFIG = [
    # --- Local Ollama Models ---
    {
        "real_name": "aminadaven/dictalm2.0-instruct:Q4_K_M",
        "clean_name": "DictaLM_2.0_Instruct_Q4",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
    {
        "real_name": "DictaLM-3.0-1.7B-Instruct:latest",
        "clean_name": "DictaLM_3.0_1.7B_Instruct",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
    {
        "real_name": "hf.co/dicta-il/DictaLM-3.0-Nemotron-12B-Instruct-GGUF:Q4_K_M",
        "clean_name": "DictaLM_3.0_Nemotron_12B_Q4",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
    {
        "real_name": "gemma3:4b",
        "clean_name": "Gemma_3_4B",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
    {
        "real_name": "qwen3:8b",
        "clean_name": "Qwen_3_8B",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
    {
        "real_name": "llama3.2:latest",
        "clean_name": "Llama_3_2",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
    {
        "real_name": "hebatron:30b-q4",
        "clean_name": "Hebatron_30B_Q4",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },

    # --- Gemini API ---
    {
        "real_name": "gemini-3.8-flash",
        "clean_name": "Gemini_3.8_Flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_KEY,
    },
    {
        "real_name": "gemini-3.1-pro-preview",
        "clean_name": "Gemini_3.1_Pro",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_KEY,
    },
    {
        "real_name": "gemma-4-31b-it",
        "clean_name": "Gemma_4_31B",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_KEY,
    },
    {
        "real_name": "gemini-3-flash-preview",
        "clean_name": "Gemini_3_Flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_KEY,
    },
    {
        "real_name": "gemini-3.5-flash-lite",
        "clean_name": "Gemini_3.5_Flash_Lite",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_KEY,
    },
    {
        "real_name": "gemma-4-26b-a4b-it",
        "clean_name": "Gemma_4_26B",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_KEY,
    },

    # --- OpenRouter API ---
    {
        "real_name": "openai/gpt-5.6-luna",
        "clean_name": "GPT_5.6_Luna",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "openai/gpt-5",
        "clean_name": "GPT_5",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "mistralai/mistral-large-2407",
        "clean_name": "Mistral_Large",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "meta-llama/llama-3.3-70b-instruct",
        "clean_name": "Llama_3.3_70B",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "qwen/qwen-2.5-72b-instruct",
        "clean_name": "Qwen_2.5_72B",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "cohere/command-a",
        "clean_name": "Cohere_Command_A",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "deepseek/deepseek-v4-flash",
        "clean_name": "DeepSeek_V4_Flash",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "deepseek/deepseek-v4.1-flash",
        "clean_name": "DeepSeek_V4.1_Flash",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "xiaomi/mimo-v2.5",
        "clean_name": "Xiaomi_Mimo_2.5",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "meta/muse-spark-1.3-contributor",
        "clean_name": "Muse_Spark_1.3",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "cohere/command-r-08-2024",
        "clean_name": "Command_R_35B",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_KEY,
    },
    {
        "real_name": "qwen/qwen-2.5-coder-32b-instruct",
        "clean_name": "Qwen_2.5_32B_Instruct",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_KEY,
    },
]

# Derived set of all configured clean names for evaluation and filtering
READY_MODELS: set[str] = {m["clean_name"] for m in MODELS_CONFIG}
