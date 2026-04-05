import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

VENDORS = ["Amazon", "Flipkart", "Reliance", "Tata Motors", "Infosys", "UnknownCorp", "ShadySupplies", "ABC Traders"]

def generate_invoice(is_fraud=False):
    invoice_id = f"INV{random.randint(1000, 9999)}"
    vendor = random.choice(VENDORS)
    amount = round(random.uniform(500, 150000), 2)
    date = (datetime.now() - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d")
    
    # Fraud injection
    if is_fraud:
        fraud_type = random.choice(["high_amount", "unknown_vendor", "duplicate"])
        if fraud_type == "high_amount":
            amount = round(random.uniform(120000, 500000), 2)
        elif fraud_type == "unknown_vendor":
            vendor = random.choice(["UnknownCorp", "ShadySupplies", "GhostVendors Ltd"])
        elif fraud_type == "duplicate":
            invoice_id = "INV4567"  # Fixed duplicate for demo
    
    text = f"""
Invoice ID: {invoice_id}
Vendor: {vendor}
Amount: {amount}
Date: {date}
Description: Services rendered for Q1 2026.
    """.strip()
    
    ground_truth = {
        "invoice_id": invoice_id,
        "vendor": vendor,
        "amount": amount,
        "is_fraud": is_fraud or (amount > 100000 or vendor in ["UnknownCorp", "ShadySupplies"])
    }
    
    return text, ground_truth

def generate_dataset(n=80):
    tasks = []
    for i in range(n):
        is_fraud = random.random() < 0.25  # 25% fraud rate
        invoice_text, gt = generate_invoice(is_fraud)
        tasks.append({
            "id": i,
            "invoice_text": invoice_text,
            "ground_truth": gt
        })
    return tasks