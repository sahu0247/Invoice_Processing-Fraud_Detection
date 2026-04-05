from agent import InvoiceAgent

# Initialize the agent (same as before)
agent = InvoiceAgent()

def predict_invoice(invoice_text: str):
    print("📄 Input Invoice:\n")
    print(invoice_text)
    print("-" * 60)
    
    # Agent makes decision step by step
    action = agent.act({"invoice_text": invoice_text})
    
    extracted = action["extracted"]
    fraud_detected = action["fraud_detected"]
    reasons = action["fraud_reasons"]
    decision = action["decision"]
    
    # Pretty Output
    print("🔍 Step 1: Extracted Fields")
    print(f"   Invoice ID : {extracted['invoice_id']}")
    print(f"   Vendor     : {extracted['vendor']}")
    print(f"   Amount     : ₹{extracted['amount']:,.2f}")
    print()
    
    print("🚨 Step 2: Fraud Analysis")
    if fraud_detected:
        print("   ❌ Fraud Detected!")
        for reason in reasons:
            print(f"      • {reason}")
    else:
        print("   ✅ No fraud signals found")
    print()
    
    print("🎯 Final Decision")
    if decision == "approve":
        print("   ✅ APPROVED")
    else:
        print("   🚩 FLAGGED for review")
    
    print("=" * 60)
    return decision, fraud_detected


# ==================== Test with some random invoices ====================

if __name__ == "__main__":
    print("🧠 Invoice Fraud Detection Agent\n")
    
    # Example 1: Normal Invoice
    normal_invoice = """
Invoice ID: INV8923
Vendor: Amazon
Amount: 12500
Date: 2026-03-15
Description: Office supplies
    """
    
    predict_invoice(normal_invoice)
    
    # Example 2: High Amount Fraud
    high_amount = """
Invoice ID: INV4455
Vendor: Reliance
Amount: 245000
Date: 2026-04-01
    """
    
    predict_invoice(high_amount)
    
    # Example 3: Unknown Vendor Fraud
    suspicious = """
Invoice ID: INV7777
Vendor: ShadySupplies
Amount: 8500
Date: 2026-03-20
    """
    
    predict_invoice(suspicious)
    
    # Example 4: Your own custom invoice
    print("\n💡 Now testing your own invoice...\n")
    custom_text = """
Invoice ID: INV9999
Vendor: UnknownCorp
Amount: 150000
Date: 2026-04-05
    """
    
    predict_invoice(custom_text)