import streamlit as st
import time
import sqlite3
import json
import pandas as pd
import plotly.express as px
from datetime import datetime

from agent import InvoiceAgent
from tasks import generate_dataset
from env import InvoiceEnv

# ========================= DATABASE SETUP =========================
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
               1 if record.get('fraud_detected', False) else 0,
               1 if record.get('is_duplicate', False) else 0,
               record.get('confidence', 80),
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

# ========================= STREAMLIT APP =========================
st.set_page_config(page_title="InvoiceGuard AI", page_icon="🛡️", layout="wide")

st.title("🛡️ InvoiceGuard AI")
st.markdown("**Store-Ready Invoice Fraud Detection System with Analytics**")

if "agent" not in st.session_state:
    st.session_state.agent = InvoiceAgent()

agent = st.session_state.agent

if "history" not in st.session_state:
    st.session_state.history = load_history()

if "tasks" not in st.session_state:
    st.session_state.tasks = None

if "show_dataset" not in st.session_state:
    st.session_state.show_dataset = False

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.title("InvoiceGuard")
    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "📤 Predict Invoice",
        "🧪 Train & Evaluate",
        "📊 History",
        "⚙️ Settings"
    ])
    st.divider()
    st.metric("Total Records", len(st.session_state.history))

# ========================= DASHBOARD WITH CHARTS + FILTERED TABLES =========================
if page == "🏠 Dashboard":
    st.header("📊 Dashboard & Analytics")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Known Vendors", len(agent.known_vendors))
    with col2: st.metric("High Amount Threshold", f"₹{agent.threshold:,}")
    with col3: st.metric("Total Invoices", len(st.session_state.history))

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        # Side-by-side Charts
        st.subheader("📈 Analytics Overview")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Fraud vs Safe Invoices")
            fraud_count = df['fraud_detected'].value_counts()
            fig_pie = px.pie(
                values=[fraud_count.get(False, 0), fraud_count.get(True, 0)],
                names=['Safe', 'Fraud'],
                color_discrete_sequence=["#22c55e", "#ef4444"],
                hole=0.45
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            st.subheader("Top Vendors by Count")
            vendor_counts = df['vendor'].value_counts().head(10)
            fig_bar = px.bar(
                x=vendor_counts.index,
                y=vendor_counts.values,
                labels={'x': 'Vendor', 'y': 'Number of Invoices'},
                color=vendor_counts.values,
                color_continuous_scale="blues"
            )
            fig_bar.update_layout(xaxis_tickangle=-45, height=450)
            st.plotly_chart(fig_bar, use_container_width=True)

        # === NEW FEATURE: View Approved / Flagged Tables ===
        st.subheader("📋 View Invoices by Status")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("✅ View Approved Invoices", use_container_width=True):
                st.session_state.view_mode = "approved"
        with col_b:
            if st.button("🚩 View Flagged Invoices", use_container_width=True):
                st.session_state.view_mode = "flagged"
        with col_c:
            if st.button("📊 View All Invoices", use_container_width=True):
                st.session_state.view_mode = "all"

        # Display Table based on selection
        if st.session_state.get("view_mode") == "approved":
            approved_df = df[df['decision'] == 'approve']
            st.subheader("✅ Approved Invoices")
            st.dataframe(approved_df, use_container_width=True, height=400)
            csv_approved = approved_df.to_csv(index=False)
            st.download_button("📥 Download Approved CSV", csv_approved, "approved_invoices.csv", "text/csv", use_container_width=True)

        elif st.session_state.get("view_mode") == "flagged":
            flagged_df = df[df['decision'] == 'flag']
            st.subheader("🚩 Flagged Invoices")
            st.dataframe(flagged_df, use_container_width=True, height=400)
            csv_flagged = flagged_df.to_csv(index=False)
            st.download_button("📥 Download Flagged CSV", csv_flagged, "flagged_invoices.csv", "text/csv", use_container_width=True)

        elif st.session_state.get("view_mode") == "all":
            st.subheader("📊 All Invoices")
            st.dataframe(df, use_container_width=True, height=400)
            csv_all = df.to_csv(index=False)
            st.download_button("📥 Download All Data as CSV", csv_all, "all_invoices.csv", "text/csv", use_container_width=True)

    else:
        st.info("No data available yet. Start analyzing invoices in the Predict section.")

# ========================= PREDICT INVOICE =========================
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
            st.error("Please provide invoice text.")
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

            st.success("✅ Analysis Complete!")

            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("📋 Extracted Fields")
                st.json(extracted)
            with col2:
                st.subheader("🚨 Analysis")
                if is_duplicate: st.error("🔴 DUPLICATE")
                if fraud_detected: st.error("❌ Fraud Detected")
                for r in reasons: st.write(f"• {r}")
                st.metric("Confidence", f"{confidence}%")

            st.subheader("🎯 Final Decision")
            if decision == "approve":
                st.success("✅ **APPROVED**")
            else:
                st.error("🚩 **FLAGGED**")

            report = f"""InvoiceGuard AI Report
Timestamp     : {record['timestamp']}
Invoice ID    : {extracted.get('invoice_id')}
Vendor        : {extracted.get('vendor')}
Amount        : ₹{extracted.get('amount', 0):,.2f}
Decision      : {decision.upper()}
Fraud         : {'Yes' if fraud_detected else 'No'}
Duplicate     : {'Yes' if is_duplicate else 'No'}
Confidence    : {confidence}%
"""
            st.download_button("📥 Download Report", report, "invoice_report.txt")

# ========================= TRAIN & EVALUATE =========================
elif page == "🧪 Train & Evaluate":
    st.header("🧪 Train & Evaluate Agent")

    if st.button("Generate 100 Synthetic Invoices", use_container_width=True):
        with st.spinner("Generating dataset..."):
            st.session_state.tasks = generate_dataset(n=100)
            st.session_state.show_dataset = True
        st.success(f"✅ Generated {len(st.session_state.tasks)} invoices")

    if st.session_state.get("show_dataset", False) and st.session_state.get("tasks"):
        if st.button("👀 View Generated Dataset", use_container_width=True):
            st.session_state.viewing_dataset = True

    if st.session_state.get("viewing_dataset", False) and st.session_state.get("tasks"):
        st.subheader("Generated Dataset Preview")
        df_tasks = pd.DataFrame([
            {
                "ID": t["id"],
                "Invoice ID": t["ground_truth"]["invoice_id"],
                "Vendor": t["ground_truth"]["vendor"],
                "Amount": round(t["ground_truth"]["amount"], 2),
                "Is Fraud": t["ground_truth"]["is_fraud"]
            } for t in st.session_state.tasks
        ])
        st.dataframe(df_tasks, use_container_width=True, height=400)

        csv = df_tasks.to_csv(index=False)
        st.download_button("📥 Download Dataset as CSV", csv, "synthetic_invoices.csv", "text/csv", use_container_width=True)

        if st.button("Close Dataset View", use_container_width=True):
            st.session_state.viewing_dataset = False

    if st.button("Run Full Evaluation", type="primary", use_container_width=True):
        if "tasks" not in st.session_state or not st.session_state.tasks:
            st.error("❌ Please generate the dataset first!")
        else:
            with st.spinner("Running evaluation..."):
                env = InvoiceEnv(st.session_state.tasks)
                obs = env.reset()
                total_reward = 0.0
                progress_bar = st.progress(0)

                for i in range(len(st.session_state.tasks)):
                    action = agent.act(obs)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    progress_bar.progress((i + 1) / len(st.session_state.tasks))

                final_info = info if isinstance(info, dict) else {}

                st.success("✅ Evaluation Completed Successfully!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reward", f"{total_reward:.2f} / 100.0")
                with col2:
                    st.metric("Overall Accuracy", f"{final_info.get('accuracy', 0):.1f}%")
                with col3:
                    st.metric("Fraud Detection Rate", f"{final_info.get('fraud_detection_rate', 0):.1f}%")

                st.metric("Extraction Success Rate", f"{final_info.get('extraction_success_rate', 0):.1f}%")

# ========================= HISTORY =========================
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
            st.success("History cleared permanently!")
            st.rerun()
    else:
        st.info("No records found yet.")

# ========================= SETTINGS =========================
elif page == "⚙️ Settings":
    st.header("⚙️ Settings")
    
    vendors_input = st.text_input("Known Vendors (comma separated)", 
                                  ", ".join(sorted(agent.known_vendors)))
    if st.button("Update Known Vendors", use_container_width=True):
        agent.known_vendors = {v.strip() for v in vendors_input.split(",") if v.strip()}
        st.success("✅ Known vendors updated!")

    new_threshold = st.number_input("High Amount Threshold (₹)", 
                                    min_value=50000, 
                                    value=agent.threshold, 
                                    step=10000)
    if st.button("Update High Amount Threshold", use_container_width=True):
        agent.threshold = new_threshold
        st.success(f"✅ Threshold updated to ₹{new_threshold:,}")

    if st.button("Clear Duplicate Memory", use_container_width=True):
        agent.reset_memory()
        st.success("Duplicate memory cleared!")

st.divider()
st.caption("InvoiceGuard AI •")
