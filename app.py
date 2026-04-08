import os
from typing import List

import gradio as gr

from inference import run_benchmark


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
        scores = run_benchmark(
            task_names=task_list,
            api_key=key,
            model_name=model_name.strip() or "gpt-4o",
            api_base_url=api_base_url.strip() or "https://api.openai.com/v1",
        )
    except Exception as exc:
        return f"Run failed: {exc}"

    lines = ["Evaluation complete.", ""]
    for name in task_list:
        lines.append(f"{name}: {scores.get(name, 0.0):.2f}/1.00")
    if len(task_list) > 1:
        avg = sum(scores.values()) / len(scores)
        lines.append("")
        lines.append(f"average: {avg:.2f}/1.00")
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


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
