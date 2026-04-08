from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from agent import InvoiceAgent
from env import InvoiceEnv
from tasks import generate_dataset

app = FastAPI(title="InvoiceGuard OpenEnv")

# Allow all requests (important)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = InvoiceAgent()
env = None
current_tasks = None


# ✅ RESET (must be SIMPLE)
@app.post("/reset")
def reset():
    global env, current_tasks
    current_tasks = generate_dataset(n=30)
    env = InvoiceEnv(current_tasks)
    obs = env.reset()

    return {
        "observation": obs
    }


# ✅ STEP (accept ANY dict)
@app.post("/step")
def step(action: Dict[str, Any]):
    global env

    if env is None:
        return {
            "observation": {},
            "reward": 0.0,
            "done": True,
            "info": {"error": "Call /reset first"}
        }

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }


# ✅ HEALTH (optional but safe)
@app.get("/health")
def health():
    return {"status": "ok"}
