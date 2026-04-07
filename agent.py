import re
from datetime import datetime
from typing import Dict, List, Any

class InvoiceAgent:
    def __init__(self):
        self.known_vendors = {"Amazon", "Flipkart", "Reliance", "Tata Motors", "Infosys", "BigBasket", "JioMart"}
        self.seen_invoice_ids = set()
        self.threshold = 100000

    def extract_fields(self, text: str) -> Dict:
        """Highly robust extraction for messy / OCR output"""
        text = text.replace("\n", " ").strip()

        # 1. Invoice ID - very flexible
        invoice_id = re.search(r"Invoice\s*(?:ID|No|Number|#)?[:\s]*(\S+)", text, re.I)

        # 2. Vendor - try multiple patterns
        vendor = None
        vendor_match = re.search(r"(?:Vendor|Seller|Supplier|From)[:\s]*(.+?)(?=\s*(?:Amount|Total|Date|Invoice ID|\d{1,2}[/-]))", text, re.I)
        if vendor_match:
            vendor = vendor_match.group(1).strip()
        else:
            # Fallback: look for known vendors in text
            for known in self.known_vendors:
                if known.lower() in text.lower():
                    vendor = known
                    break

        # 3. Amount - improved
        amount_match = re.search(r"(?:Amount|Total|Grand Total|Payable|Net)[:\s]*[₹$]?\s*([\d,]+\.?\d*)", text, re.I)
        amount = 0.0
        if amount_match:
            amount_str = amount_match.group(1).replace(",", "").strip()
            try:
                amount = float(amount_str)
            except ValueError:
                amount = 0.0

        # 4. Date
        date_match = re.search(r"(?:Date|Invoice Date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text, re.I)

        return {
            "invoice_id": invoice_id.group(1) if invoice_id else None,
            "vendor": vendor,
            "amount": amount,
            "date": date_match.group(1) if date_match else None
        }

    def check_fraud(self, extracted: Dict, is_duplicate: bool = False) -> tuple:
        flags = []
        confidence = 80

        amount = extracted.get("amount", 0)
        vendor = extracted.get("vendor")
        invoice_id = extracted.get("invoice_id")

        # High Amount
        if amount > self.threshold:
            flags.append(f"High amount (₹{amount:,.2f})")
            confidence -= 20

        # Unknown Vendor
        if vendor:
            vendor_clean = vendor.strip()
            is_known = any(v.lower() in vendor_clean.lower() for v in self.known_vendors)
            if not is_known:
                flags.append(f"Unknown vendor: {vendor_clean}")
                confidence -= 25
        else:
            flags.append("Missing Vendor")
            confidence -= 15

        # Duplicate
        if is_duplicate:
            flags.append("Duplicate Invoice ID")
            confidence -= 30

        # Missing Invoice ID
        if not invoice_id:
            flags.append("Missing Invoice ID")
            confidence -= 20

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