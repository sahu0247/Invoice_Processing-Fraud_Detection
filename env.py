class InvoiceEnv:
    def __init__(self, tasks):
        self.tasks = tasks
        self.current_task_idx = 0
        self.state = {}
        self.total_reward = 0.0
        self.done = False
    
    def reset(self):
        self.current_task_idx = 0
        self.total_reward = 0.0
        self.done = False
        return self._get_observation()
    
    def _get_observation(self):
        if self.current_task_idx >= len(self.tasks):
            self.done = True
            return None
        task = self.tasks[self.current_task_idx]
        return {
            "invoice_text": task["invoice_text"],
            "step": "start"
        }
    
    def step(self, action):
        """action: dict with keys like 'extracted', 'fraud_check', 'decision'"""
        task = self.tasks[self.current_task_idx]
        gt = task["ground_truth"]
        reward = 0.0
        
        # Extraction reward
        if "extracted" in action:
            ext = action["extracted"]
            if (ext.get("invoice_id") == gt["invoice_id"] and
                abs(ext.get("amount", 0) - gt["amount"]) < 1.0 and
                ext.get("vendor") == gt["vendor"]):
                reward += 0.2
        
        # Fraud check reward
        if "fraud_detected" in action:
            predicted_fraud = action["fraud_detected"]
            if predicted_fraud == gt["is_fraud"]:
                reward += 0.3
        
        # Final decision reward
        if "decision" in action:
            decision = action["decision"]  # "approve" or "flag"
            correct = (decision == "flag" and gt["is_fraud"]) or (decision == "approve" and not gt["is_fraud"])
            if correct:
                reward += 0.5
            else:
                reward -= 0.5
        
        self.total_reward += reward
        self.current_task_idx += 1
        obs = self._get_observation()
        
        return obs, reward, self.done, {"total_reward": self.total_reward}