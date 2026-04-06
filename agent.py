import re
from datetime import datetime
from typing import Dict, List, Any

class InvoiceAgent:
    def __init__(self):
        self.known_vendors = {"Amazon", "Flipkart", "Reliance", "Tata Motors", "Infosys", "BigBasket", "JioMart"}
        self.seen_invoice_ids = set()
        self.threshold = 100000

    def extract_fields(self, text: str) -> Dict:
        """Robust field extraction for real and synthetic invoices"""
        # Invoice ID
        invoice_id = re.search(r"Invoice\s*(?:ID|No|Number|#)[:\s]*(\S+)", text, re.I)
        
        # Vendor / Seller
        vendor = re.search(r"(?:Vendor|Seller|Supplier)[:\s]*(.+?)(?:\n|$)", text, re.I)
        
        # Amount (improved regex to handle different formats)
        amount_match = re.search(r"(?:Amount|Total|Grand Total|Payable|Net Amount)[:\s]*[₹$]?\s*([\d,]+\.?\d*)", text, re.I)
        
        # Date
        date = re.search(r"(?:Date|Invoice Date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text, re.I)

        # Safe amount extraction
        if amount_match:
            amount_str = amount_match.group(1).replace(",", "").strip()
            try:
                amount = float(amount_str)
            except ValueError:
                amount = 0.0
        else:
            amount = 0.0

        return {
            "invoice_id": invoice_id.group(1) if invoice_id else None,
            "vendor": vendor.group(1).strip() if vendor else None,
            "amount": amount,
            "date": date.group(1) if date else None
        }

    def check_fraud(self, extracted: Dict, is_duplicate: bool = False) -> tuple:
        flags = []
        confidence = 85

        amount = extracted.get("amount", 0)

        # High Amount
        if amount > self.threshold:
            flags.append(f"High amount (₹{amount:,.2f})")
            confidence -= 20

        # Unknown Vendor
        vendor = extracted.get("vendor")
        if vendor and vendor not in self.known_vendors:
            flags.append(f"Unknown vendor: {vendor}")
            confidence -= 25

        # Duplicate
        if is_duplicate:
            flags.append("Duplicate Invoice ID")
            confidence -= 30

        # Missing critical fields
        if not extracted.get("invoice_id"):
            flags.append("Missing Invoice ID")
            confidence -= 20
        if not vendor:
            flags.append("Missing Vendor")
            confidence -= 15

        # Suspicious round amount
        if amount > 50000 and amount % 1000 == 0:
            flags.append("Suspicious round amount")
            confidence -= 10

        is_fraud = len(flags) > 0
        return is_fraud, flags, max(40, confidence)

    def decide(self, is_fraud: bool) -> str:
        return "flag" if is_fraud else "approve"

    def act(self, observation: Dict) -> Dict[str, Any]:
        text = observation.get("invoice_text", "")

        extracted = self.extract_fields(text)
        invoice_id = extracted.get("invoice_id")
        is_duplicate = bool(invoice_id and invoice_id in self.seen_invoice_ids)

        fraud_detected, reasons, confidence = self.check_fraud(extracted, is_duplicate)
        decision = self.decide(fraud_detected)

        # Update memory
        if invoice_id:
            self.seen_invoice_ids.add(invoice_id)

        return {
            "extracted": extracted,
            "fraud_detected": fraud_detected,
            "fraud_reasons": reasons,
            "decision": decision,
            "is_duplicate": is_duplicate,
            "confidence": confidence,
            "memory_size": len(self.seen_invoice_ids)
        }

    def reset_memory(self):
        self.seen_invoice_ids.clear()