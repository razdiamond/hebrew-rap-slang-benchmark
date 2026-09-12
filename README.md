# `hebrew-rap-slang-benchmark`

An NLP evaluation benchmark for assessing cultural metaphors, figurative language, and slang comprehension in Hebrew rap lyrics across Large Language Models (LLMs).

---

## 📌 Overview

While state-of-the-art (SOTA) general LLMs and Hebrew-specialized models achieve strong performance on standard, literal Hebrew text, they frequently fail when processing deep cultural context, local idioms, and modern slang. Israeli hip-hop and rap lyrics are dense with figurative expressions and subcultural references that cannot be interpreted literally.

**`hebrew-rap-slang-benchmark`** provides a hand-curated evaluation suite of ~120–150 Hebrew rap lyric snippets paired with crowd-verified Ground Truth (GT) explanations from Genius. Models are evaluated across two complementary tasks:

1. **Multiple-Choice QA:** Selecting the correct cultural meaning out of 4 options (Accuracy %).
2. **Open-Ended Explanation:** Generating a free-text explanation, evaluated using:
   - **Hebrew SBERT Cosine Similarity** against the Genius Ground Truth.
   - **LLM-as-a-Judge Scoring** (1–5 scale evaluating cultural nuance and accuracy).

---

## 📁 Repository Structure

```text
hebrew-rap-slang-benchmark/
├── data/
│   └── hebrew_rap_slang.json       # Hand-curated dataset (lyrics, GT, options)
├── results/
│   └── raw_responses.json          # Batch inference logs across models & prompts
├── src/
│   ├── scraper.py                  # Genius scraping & curation tools
│   ├── pipeline.py                 # Main execution script (Ollama + APIs)
│   └── metrics.py                  # SBERT cosine similarity & LLM-as-a-Judge
├── analysis.ipynb                  # Jupyter notebook for metrics & visualization
├── .env.example                    # Template for API keys (Gemini, OpenAI, Grok)
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Pinned dependencies
└── README.md                       # Project documentation
```

---

## 🛠️ Environment & Hardware

- **OS:** Windows 11
- **Python Version:** 3.12
- **Hardware (Local Testing):** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
- **Local LLM Runtime:** [Ollama](https://ollama.com/) (running quantized 7B/8B models for offline debugging)

---

## 🚀 Installation & Setup

### 1. Clone Repository & Set Up Virtual Environment

```bash
git clone https://github.com/razdiamond/hebrew-rap-slang-benchmark.git
cd hebrew-rap-slang-benchmark

python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY="your_gemini_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
GROK_API_KEY="your_grok_api_key_here"
GENIUS_ACCESS_TOKEN="your_genius_token_here"
```

---

## 🧪 Running the Pipeline

### 1. Local Testing (Ollama)
Ensure Ollama is running locally on port `11434`, then execute the pipeline script:

```bash
python src/pipeline.py --provider local --model llama3.2
```

### 2. Full API Benchmark Run
Execute batch inference across SOTA models with Zero-Shot (ZS), Few-Shot (FS), and Chain-of-Thought (CoT) prompting:

```bash
python src/pipeline.py --provider gemini --model gemini-1.5-pro
python src/pipeline.py --provider openai --model gpt-4o
```

### 3. Metric Computation & Analysis
Launch the analysis notebook to compute MC Accuracy, Hebrew SBERT cosine similarity, and generate publication plots:

```bash
jupyter lab analysis.ipynb
```

---

## 📐 Evaluation Methodology

Each model prompt requests a structured JSON response containing both tasks in a single turn:

- **Zero-Shot (ZS):** Direct prompt asking for explanation + option selection.
- **Few-Shot (FS):** 2–3 in-context Hebrew rap slang examples provided.
- **Chain-of-Thought (CoT):** Step-by-step reasoning in Hebrew prior to outputting final answer.

---

## 👤 Author & Course Information

- **Author:** Raz Diamond
- **Email:** `raz.diamond@post.runi.ac.il`
- **Institution:** Reichman University (RUNI), Efi Arazi School of Computer Science
- **Course:** Natural Language Processing (NLP), 2026