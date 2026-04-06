class InvoiceEnv:
    def __init__(self, tasks):
        self.tasks = tasks
        self.current_task_idx = 0
        self.total_reward = 0.0
        self.done = False
        self.stats = {
            "correct_decisions": 0,
            "total_tasks": 0,
            "fraud_correct": 0,
            "total_fraud": 0,
            "extraction_success": 0
        }

    def reset(self):
        self.current_task_idx = 0
        self.total_reward = 0.0
        self.done = False
        self.stats = {k: 0 for k in self.stats}
        return self._get_observation()

    def _get_observation(self):
        if self.current_task_idx >= len(self.tasks):
            self.done = True
            return None
        task = self.tasks[self.current_task_idx]
        return {"invoice_text": task["invoice_text"]}

    def step(self, action):
        task = self.tasks[self.current_task_idx]
        gt = task["ground_truth"]
        reward = 0.0

        extracted = action.get("extracted", {})
        predicted_fraud = action.get("fraud_detected", False)
        actual_fraud = gt.get("is_fraud", False)

        self.stats["total_tasks"] += 1
        if actual_fraud:
            self.stats["total_fraud"] += 1

        # 1. Extraction Reward (more forgiving)
        extraction_ok = (
            extracted.get("invoice_id") == gt.get("invoice_id") and
            abs(extracted.get("amount", 0) - gt.get("amount", 0)) < 50 and  # increased tolerance
            extracted.get("vendor") == gt.get("vendor")
        )
        if extraction_ok:
            reward += 0.30
            self.stats["extraction_success"] += 1
        else:
            reward -= 0.10   # small penalty instead of big one

        # 2. Fraud Detection Reward (most important)
        if predicted_fraud == actual_fraud:
            reward += 0.50
            self.stats["correct_decisions"] += 1
            if actual_fraud:
                self.stats["fraud_correct"] += 1
        else:
            reward -= 0.40

        self.total_reward += reward
        self.current_task_idx += 1

        obs = self._get_observation()

        accuracy = self.stats["correct_decisions"] / self.stats["total_tasks"] if self.stats["total_tasks"] > 0 else 0
        fraud_rate = (self.stats["fraud_correct"] / self.stats["total_fraud"] * 100) if self.stats["total_fraud"] > 0 else 0

        return obs, reward, self.done, {
            "total_reward": round(self.total_reward, 2),
            "accuracy": round(accuracy * 100, 1),
            "fraud_detection_rate": round(fraud_rate, 1),
            "extraction_success_rate": round((self.stats["extraction_success"] / self.stats["total_tasks"] * 100), 1) if self.stats["total_tasks"] > 0 else 0
        }