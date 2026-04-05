# 🧠 AI Invoice Fraud Detection Agent

An **OpenEnv-style AI Agent** that reads raw invoice text, extracts important fields, detects fraud, and decides whether to **Approve** or **Flag** the invoice.

This project demonstrates a complete agentic workflow with synthetic data, modular design, reward-based evaluation, and a user-friendly Streamlit web interface.

---

## ✨ Key Features

- Synthetic dataset generation (50–100 invoices automatically)
- Step-by-step agent reasoning (Extract → Fraud Check → Decide)
- Fraud detection based on amount and vendor
- OpenEnv-style environment with observation, action & reward
- Interactive web demo using Streamlit
- Fully offline – no APIs needed

---

## 📁 Project Structure & File Explanation

| File              | Description |
|-------------------|-----------|
| **`baseline.py`**     | Main script to run the full evaluation. Generates 100 invoices, runs the environment + agent on all tasks, and shows the final score (total reward). |
| **`tasks.py`**        | Responsible for creating synthetic invoices. Uses Faker to generate realistic invoice text with random vendors, amounts, and injects fraud cases (high amount, unknown vendors, etc.). Returns tasks with ground truth. |
| **`env.py`**          | OpenEnv-style simulator (like a game environment). It gives observations to the agent, receives actions, calculates rewards, and tracks progress across tasks. |
| **`agent.py`**        | The brain of the project. Contains the `InvoiceAgent` class that performs three main steps: extracts fields using regex, checks for fraud using rules, and makes the final approve/flag decision. |
| **`predict.py`**      | Simple script to test the agent on individual or custom invoices. Good for quick testing and debugging. |
| **`app.py`**          | Streamlit web application. Provides a clean, user-friendly interface where anyone can paste an invoice and see the agent's step-by-step reasoning and final decision. |
| **`requirements.txt`** | List of Python packages needed to run the project (`faker` and `streamlit`). |
| **`README.md`**       | This file – project documentation. |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install faker streamlit