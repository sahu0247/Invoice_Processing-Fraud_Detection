import streamlit as st
import time
import sqlite3
import json
from datetime import datetime
from agent import InvoiceAgent
from tasks import generate_dataset
from env import InvoiceEnv

# Database setup (keep as before)
def init_db():
    conn = sqlite3.connect('invoices.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    invoice_id TEXT,
                    vendor TEXT,
                    amount REAL,
                    decision TEXT,
                    fraud_detected INTEGER,
                    is_duplicate INTEGER,
                    confidence INTEGER,
                    reasons TEXT
                )''')
    conn.commit()
    conn.close()

def save_to_db(record):
    conn = sqlite3.connect('invoices.db')
    c = conn.cursor()
    c.execute('''INSERT INTO history 
                (timestamp, invoice_id, vendor, amount, decision, fraud_detected, is_duplicate, confidence, reasons)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (record['timestamp'], record['invoice_id'], record['vendor'], 
               record['amount'], record['decision'], 
               1 if record['fraud_detected'] else 0,
               1 if record['is_duplicate'] else 0,
               record['confidence'],
               json.dumps(record.get('reasons', []))))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect('invoices.db')
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    history = []
    for row in rows:
        history.append({
            "timestamp": row[1],
            "invoice_id": row[2],
            "vendor": row[3],
            "amount": row[4],
            "decision": row[5],
            "fraud_detected": bool(row[6]),
            "is_duplicate": bool(row[7]),
            "confidence": row[8],
            "reasons": json.loads(row[9]) if row[9] else []
        })
    return history

init_db()

st.set_page_config(page_title="InvoiceGuard AI", page_icon="🛡️", layout="wide")

st.title("🛡️ InvoiceGuard AI")
st.markdown("**Store-Ready Invoice Fraud Detection System**")

if "agent" not in st.session_state:
    st.session_state.agent = InvoiceAgent()

agent = st.session_state.agent

if "history" not in st.session_state:
    st.session_state.history = load_history()

with st.sidebar:
    st.title("InvoiceGuard")
    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "📤 Predict Invoice",
        "🧪 Train & Evaluate",
        "📊 History",
        "⚙️ Settings"
    ])
    st.metric("Total Records", len(st.session_state.history))

# Dashboard (same as before)
if page == "🏠 Dashboard":
    st.header("Welcome to InvoiceGuard AI")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Known Vendors", len(agent.known_vendors))
    with col2: st.metric("High Amount Threshold", f"₹{agent.threshold:,}")
    with col3: st.metric("Total Invoices", len(st.session_state.history))
    st.success("✅ Database + Improved Agent Active")

# Predict Invoice (same as before - shortened for space)
elif page == "📤 Predict Invoice":
    st.header("📤 Predict Invoice")
    uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
    invoice_text = ""

    if uploaded_file:
        invoice_text = uploaded_file.getvalue().decode("utf-8")
        st.success("✅ Text file loaded")

    if not invoice_text:
        invoice_text = st.text_area("Or paste invoice text here:", height=300)

    if st.button("🚀 Analyze Invoice", type="primary", use_container_width=True):
        if not invoice_text.strip():
            st.error("Please provide text.")
        else:
            with st.spinner("Analyzing..."):
                result = agent.act({"invoice_text": invoice_text})

            extracted = result["extracted"]
            fraud_detected = result["fraud_detected"]
            reasons = result["fraud_reasons"]
            decision = result["decision"]
            is_duplicate = result.get("is_duplicate", False)
            confidence = result.get("confidence", 80)

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "invoice_id": extracted.get("invoice_id"),
                "vendor": extracted.get("vendor"),
                "amount": extracted.get("amount", 0),
                "decision": decision,
                "fraud_detected": fraud_detected,
                "is_duplicate": is_duplicate,
                "confidence": confidence,
                "reasons": reasons
            }
            st.session_state.history.append(record)
            save_to_db(record)

            st.success("Analysis Complete!")
            col1, col2 = st.columns([1,2])
            with col1: st.json(extracted)
            with col2:
                if is_duplicate: st.error("🔴 DUPLICATE")
                if fraud_detected: st.error("❌ Fraud Detected")
                for r in reasons: st.write(f"• {r}")
            st.metric("Confidence", f"{confidence}%")
            if decision == "approve":
                st.success("✅ APPROVED")
            else:
                st.error("🚩 FLAGGED")

# ========================= TRAIN & EVALUATE =========================
elif page == "🧪 Train & Evaluate":
    st.header("🧪 Train & Evaluate Agent")

    if st.button("Generate 100 Synthetic Invoices", use_container_width=True):
        with st.spinner("Generating dataset..."):
            st.session_state.tasks = generate_dataset(n=100)
        st.success(f"✅ Generated {len(st.session_state.tasks)} invoices")

    if st.button("Run Full Evaluation", type="primary", use_container_width=True):
        if "tasks" not in st.session_state or not st.session_state.tasks:
            st.error("Please generate the dataset first!")
        else:
            with st.spinner("Running improved evaluation..."):
                env = InvoiceEnv(st.session_state.tasks)
                obs = env.reset()
                total_reward = 0.0
                progress_bar = st.progress(0)

                for i in range(len(st.session_state.tasks)):
                    action = agent.act(obs)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    progress_bar.progress((i + 1) / len(st.session_state.tasks))

                st.success("✅ Evaluation Completed Successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reward", f"{info.get('total_reward', 0)} / 100.0")
                with col2:
                    st.metric("Overall Accuracy", f"{info.get('accuracy', 0)}%")
                with col3:
                    st.metric("Fraud Detection Rate", f"{info.get('fraud_detection_rate', 0)}%")

                st.metric("Extraction Success Rate", f"{info.get('extraction_success_rate', 0)}%")
# History and Settings (same as previous)
elif page == "📊 History":
    st.header("📊 Processing History")
    if st.session_state.history:
        st.dataframe(st.session_state.history, use_container_width=True)
        if st.button("Clear All History", use_container_width=True):
            st.session_state.history = []
            conn = sqlite3.connect('invoices.db')
            conn.execute("DELETE FROM history")
            conn.commit()
            conn.close()
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No records found.")

elif page == "⚙️ Settings":
    st.header("⚙️ Settings")
    vendors_input = st.text_input("Known Vendors", ", ".join(sorted(agent.known_vendors)))
    if st.button("Update Known Vendors", use_container_width=True):
        agent.known_vendors = {v.strip() for v in vendors_input.split(",") if v.strip()}
        st.success("Updated!")

    new_threshold = st.number_input("High Amount Threshold (₹)", value=agent.threshold, step=10000)
    if st.button("Update Threshold", use_container_width=True):
        agent.threshold = new_threshold
        st.success(f"Threshold updated to ₹{new_threshold:,}")

st.divider()
st.caption("InvoiceGuard AI • SQLite Database • Fixed Train & Evaluate")