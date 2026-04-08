import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Body
import gradio as gr

from .env import CustomerSupportEnv
from .models import Action, Observation, Reward
from .inference import run_benchmark

# Initialize FastAPI app
app = FastAPI()

# Single environment instance for the API
# In a real production scenario, you might want session management,
# but for evaluation benchmarks, a single instance is standard.
env_instance = CustomerSupportEnv(task_name="easy")

@app.post("/reset")
async def reset(task_id: str = Body(default="easy", embed=True)):
    """OpenEnv compliant Reset endpoint."""
    global env_instance
    env_instance = CustomerSupportEnv(task_name=task_id)
    obs = env_instance.reset()
    return obs.model_dump()

@app.post("/step")
async def step(action: Action):
    """OpenEnv compliant Step endpoint."""
    obs, reward, done, info = env_instance.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

@app.get("/health")
async def health():
    return {"status": "ok", "env": "customer-support-ticket-triage"}

# --- Existing Gradio Evaluation Logic ---

def evaluate(task_name: str, model_name: str, api_base_url: str, api_key: str) -> str:
    key = (api_key or "").strip() or os.getenv("OPENAI_API_KEY", "")
    if not key:
        return "Missing API key. Add OPENAI_API_KEY in Hugging Face Space Secrets, or enter it in the UI."

    task_list: List[str]
    if task_name == "all":
        task_list = ["easy", "medium", "hard"]
    else:
        task_list = [task_name]

    try:
        results = run_benchmark(
            task_names=task_list,
            api_key=key,
            model_name=model_name.strip() or "gpt-4o",
            api_base_url=api_base_url.strip() or "https://api.openai.com/v1",
        )
    except Exception as exc:
        return f"Run failed: {exc}"

    lines = ["# Evaluation Results", ""]
    overall_scores = []
    for name in task_list:
        data = results.get(name, {})
        score = data.get("score", 0.0)
        trace = data.get("trace", [])
        overall_scores.append(score)
        
        lines.append(f"## Task: {name.upper()}")
        lines.append(f"**Score: {score:.2f}/1.00**")
        lines.append("### Execution Log:")
        for log_entry in trace:
            lines.append(f"- {log_entry}")
        lines.append("")

    if len(task_list) > 1:
        avg = sum(overall_scores) / len(overall_scores)
        lines.insert(2, f"**AVERAGE SCORE: {avg:.2f}/1.00**")
        lines.insert(3, "---")
        
    return "\n".join(lines)

with gr.Blocks(title="Customer Support Ticket Triage - OpenEnv") as demo:
    gr.Markdown(
        "# Customer Support Ticket Triage Environment\n"
        "Run OpenEnv task evaluations for hackathon demo submissions."
    )

    with gr.Row():
        task_name = gr.Dropdown(
            choices=["easy", "medium", "hard", "all"],
            value="all",
            label="Task",
        )
        model_name = gr.Textbox(value="gpt-4o", label="Model Name")

    api_base_url = gr.Textbox(value="https://api.openai.com/v1", label="API Base URL")
    api_key = gr.Textbox(
        value="",
        type="password",
        label="API Key (optional if OPENAI_API_KEY secret is set)",
    )
    run_btn = gr.Button("Run Evaluation", variant="primary")
    output = gr.Textbox(label="Result", lines=12)

    run_btn.click(
        fn=evaluate,
        inputs=[task_name, model_name, api_base_url, api_key],
        outputs=output,
    )

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
