from tasks import generate_dataset
from env import InvoiceEnv
from agent import InvoiceAgent

def main():
    print("🚀 Generating synthetic dataset...")
    tasks = generate_dataset(n=100)
    
    print("🏗️  Creating environment...")
    env = InvoiceEnv(tasks)
    
    print("🧠 Initializing agent...")
    agent = InvoiceAgent()
    
    print("🔄 Running episodes...\n")
    obs = env.reset()
    total_reward = 0.0
    correct_decisions = 0
    
    while not env.done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        if "decision" in action and reward > 0:  # Simplified
            correct_decisions += 1 if reward >= 0.5 else 0  # rough
        
        if done:
            break
    
    print("✅ Evaluation Complete!")
    print(f"Total Reward: {env.total_reward:.2f} / 100.0")
    print(f"Average Reward per Task: {env.total_reward / len(tasks):.3f}")
    print(f"Success Rate (approx): {correct_decisions / len(tasks) * 100:.1f}%")
    

if __name__ == "__main__":
    main()