import json
from typing import Dict, Any

from agent import InvoiceAgent

agent = InvoiceAgent()

def predict(invoice_text: str) -> Dict[str, Any]:
    """Main function expected by many OpenEnv evaluators"""
    if not invoice_text or not isinstance(invoice_text, str):
        return {
            "error": "Invalid invoice text",
            "extracted": {},
            "fraud_detected": False,
            "decision": "approve",
            "confidence": 0,
            "status": "error"
        }

    observation = {"invoice_text": invoice_text.strip()}
    result = agent.act(observation)

    extracted = result.get("extracted", {})
    
    return {
        "extracted": {
            "invoice_id": extracted.get("invoice_id"),
            "vendor": extracted.get("vendor"),
            "amount": float(extracted.get("amount", 0.0)),
            "date": extracted.get("date")
        },
        "fraud_detected": result.get("fraud_detected", False),
        "fraud_reasons": result.get("fraud_reasons", []),
        "decision": result.get("decision", "approve"),
        "is_duplicate": result.get("is_duplicate", False),
        "confidence": result.get("confidence", 80),
        "status": "success"
    }


# Local Testing
if __name__ == "__main__":
    print("🚀 Testing InvoiceGuard Inference\n")
    
    test_cases = [
        """Invoice ID: INV8923
Vendor: Amazon
Amount: 12500
Date: 2026-03-15""",
        
        """Invoice ID: INV4455
Vendor: Reliance
Amount: 245000
Date: 2026-04-01""",
        
        """Invoice ID: INV7777
Vendor: ShadySupplies
Amount: 8500
Date: 2026-03-20"""
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"=== Test {i} ===")
        result = predict(text)
        print(json.dumps(result, indent=2))
        print("-" * 60)
