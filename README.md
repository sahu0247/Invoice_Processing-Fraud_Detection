# 🧠 AI Invoice Fraud Detection Agent

**OpenEnv-style AI Agent** for automated invoice processing and fraud detection in finance workflows.

This project simulates a real-world intelligent agent that reads **unstructured/raw invoice text**, extracts key fields, applies fraud detection rules, and makes autonomous decisions to **Approve** or **Flag** invoices.

It demonstrates modern **agentic AI** concepts: step-by-step reasoning, modular actions, reward-based evaluation, and synthetic data generation — all without needing external APIs.

Perfect for learning AI agents, reinforcement learning-style environments, or building finance automation tools.

---

## 🌟 Highlights

- Fully synthetic dataset generation (no manual data required)
- Clear **OpenEnv-style** environment with observation-action-reward loop
- Modular agent with transparent step-by-step reasoning
- Reward system (max **1.0** per task) for proper extraction, fraud detection & decision
- Beautiful interactive **Streamlit web demo**
- Handles noisy and unstructured invoice text
- Lightweight, fully offline, and easy to extend
- Great baseline for real-world Accounts Payable (AP) automation

---

## 📁 Project Structure & File Explanation

| File                  | Purpose |
|-----------------------|--------|
| **`baseline.py`**         | Runs the complete experiment: generates 100 synthetic tasks, creates the environment, lets the agent process every invoice, and prints the final evaluation score (total reward and success rate). Best for benchmarking. |
| **`tasks.py`**            | Generates synthetic invoices using Faker. Creates realistic invoice text with random vendors, amounts, and dates. Also injects controlled fraud cases (high amount, unknown vendors) along with ground truth labels. Shows how to create scalable training/evaluation data. |
| **`env.py`**              | Implements an **OpenEnv-style simulator** (inspired by Gym/OpenAI environments). Manages task progression, provides observations to the agent, receives actions, calculates per-step rewards, and tracks overall performance. |
| **`agent.py`**            | Core intelligence — the `InvoiceAgent` class. Performs three key actions: field extraction (using regex), fraud checking (rule-based), and final decision making. Easy to upgrade (e.g., replace regex with LLM). |
| **`predict.py`**          | Quick testing script. Allows you to run the agent on single or custom invoices and see detailed output in the terminal. Useful for development and debugging. |
| **`app.py`**              | Streamlit-based web interface. Users can paste any invoice text and instantly see extracted fields, fraud analysis, reasoning, and the final decision in a clean UI. |
| **`requirements.txt`**    | Lists all Python dependencies (`faker` for data generation and `streamlit` for the web app). |
| **`README.md`**           | This documentation file. |

---

## 🎯 How It Works (Workflow)

The agent follows a clear **multi-step reasoning process**:

1. **Observation** — Receives raw invoice text from the environment.
2. **Extraction** — Parses Invoice ID, Vendor, and Amount using regex.
3. **Fraud Check** — Applies intelligent rules:
   - Amount > ₹1,00,000 → High-risk
   - Vendor is unknown/suspicious (e.g., `UnknownCorp`, `ShadySupplies`)
4. **Decision** — Outputs `approve` or `flag`.
5. **Reward** — Environment evaluates the action:
   - Correct extraction: **+0.2**
   - Correct fraud detection: **+0.3**
   - Correct final decision: **+0.5**
   - Wrong decision: **-0.5**
   - **Maximum score per invoice = 1.0**

This structure makes the agent transparent, debuggable, and extensible.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher

### 1. Clone the repository (or download the files)

### 2. Install dependencies


pip install -r requirements.txt

### 3. Run the project in different ways 

**A. Full Evaluation & Scoring**

Bashpython baseline.py

**B. Quick Prediction on Custom Invoices**

Bashpython predict.py

**C. Interactive Web Demo (Recommended)**

Bashstreamlit run app.py

---

### 🧪 Example Outputs

**Normal Invoice**

textInvoice ID: INV8923 | Vendor: Amazon | Amount: 12500 → ✅ APPROVED

**Fraud Cases**

High amount (₹245,000) → 🚩 FLAGGED (High amount)

Suspicious vendor (ShadySupplies) → 🚩 FLAGGED (Unknown vendor)

---

### 🛠️ Technologies Used

Python 3

Faker — Synthetic realistic data generation

Streamlit — Fast and beautiful web UI

Regex-based parsing (easily replaceable with LLMs)

Rule-based reasoning (hybrid agent design)

OpenEnv-style architecture for agent-environment interaction

---
