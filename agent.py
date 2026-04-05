import re

class InvoiceAgent:
    def __init__(self):
        self.known_vendors = {"Amazon", "Flipkart", "Reliance", "Tata Motors", "Infosys"}
    
    def extract_fields(self, invoice_text):
        # Simple regex + parsing (can be upgraded to LLM later)
        invoice_id = re.search(r"Invoice ID:\s*(\S+)", invoice_text)
        vendor = re.search(r"Vendor:\s*(.+)", invoice_text)
        amount = re.search(r"Amount:\s*([\d.]+)", invoice_text)
        
        return {
            "invoice_id": invoice_id.group(1) if invoice_id else None,
            "vendor": vendor.group(1).strip() if vendor else None,
            "amount": float(amount.group(1)) if amount else 0.0
        }
    
    def check_fraud(self, extracted):
        flags = []
        if extracted["amount"] > 100000:
            flags.append("High amount")
        if extracted["vendor"] not in self.known_vendors and extracted["vendor"] is not None:
            flags.append("Unknown vendor")
        # Duplicate check would need memory across tasks (can be added)
        
        is_fraud = len(flags) > 0
        return is_fraud, flags
    
    def decide(self, extracted, is_fraud):
        if is_fraud:
            return "flag"
        return "approve"
    
    def act(self, observation):
        text = observation["invoice_text"]
        
        # Step 1: Extract
        extracted = self.extract_fields(text)
        
        # Step 2: Fraud check
        fraud_detected, reasons = self.check_fraud(extracted)
        
        # Step 3: Decide
        decision = self.decide(extracted, fraud_detected)
        
        return {
            "extracted": extracted,
            "fraud_detected": fraud_detected,
            "fraud_reasons": reasons,
            "decision": decision
        }