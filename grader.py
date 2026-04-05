def grade(env):
    state = env.state()
    expected = env.current_task["expected"]

    score = 0.0

    if int(state["extracted_fields"].get("amount", 0)) == expected["amount"]:
        score += 0.5

    if state["fraud_detected"] == expected["fraud"]:
        score += 0.5

    return score