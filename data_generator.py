import random

vendors = ["Amazon", "Google", "Microsoft", "UnknownCorp", "Meta"]

def generate_invoice():
    invoice_id = f"INV{random.randint(100,999)}"
    vendor = random.choice(vendors)
    amount = random.choice([1000, 4500, 20000, 150000])
    date = f"2024-0{random.randint(1,9)}-{random.randint(10,28)}"

    text = f"Invoice ID: {invoice_id} | Vendor: {vendor} | Date: {date} | Amount: {amount}"

    fraud = False
    if amount > 100000 or vendor == "UnknownCorp":
        fraud = True

    return {
        "invoice_text": text,
        "ground_truth": {
            "invoice_id": invoice_id,
            "vendor": vendor,
            "amount": amount,
            "fraud": fraud
        }
    }

def generate_dataset(n=50):
    return [generate_invoice() for _ in range(n)]