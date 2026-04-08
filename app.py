from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

from agent import InvoiceAgent
from env import InvoiceEnv
from tasks import generate_dataset

app = FastAPI(title="InvoiceGuard OpenEnv")

# CORS and Proxy fix for Hugging Face Spaces + OpenEnv
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

class Action(BaseModel):
    extracted: Dict = {}
    fraud_detected: bool = False
    fraud_reasons: list = []
    decision: str = "approve"

# Reset Endpoint - Both paths to prevent 405 Method Not Allowed
@app.post("/reset")
@app.post("/")
async def reset():
    global env, current_tasks
    current_tasks = generate_dataset(n=30)
    env = InvoiceEnv(current_tasks)
    obs = env.reset()
    return {
        "observation": obs,
        "success": True,
        "task_count": len(current_tasks)
    }

@app.post("/step")
async def step(action: Action):
    global env
    if env is None:
        return {"error": "Call /reset first", "success": False}
    
    obs, reward, done, info = env.step(action.dict())
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
        "success": True
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "agent_ready": True}

@app.head("/")
@app.head("/reset")
async def head_root():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=7860,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
