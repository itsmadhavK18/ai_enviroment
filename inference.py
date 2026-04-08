import json
import re
from typing import Dict, List, Tuple, Any
from openai import OpenAI

from env import CustomerSupportEnv
from models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "dummy-key")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
MAX_STEPS = 15

SYSTEM_PROMPT = """
You are an AI customer support triage agent operating within a simulated ticket management system.
You will be presented with a Goal, a summary of Open Tickets, and the result of your last action.
When you view a ticket, you will see its full details.

Reply with EXACTLY ONE JSON object representing your action.
The action must strictly adhere to this schema:
{
    "action_type": "read_ticket" | "classify_ticket" | "assign_priority" | "draft_response" | "resolve_ticket" | "escalate_ticket" | "submit",
    "ticket_id": "T-...", 
    "issue_type": "...", // billing, technical, inquiry
    "priority": "...", // high, medium, low
    "response_text": "..." // string
}

Include ONLY the JSON object in your response. Do not include markdown code blocks or explanations.
"""

def build_user_prompt(step: int, obs) -> str:
    prompt = f"Step: {step}\n"
    prompt += f"Goal: {obs.goal}\n"
    prompt += f"Open Tickets: {', '.join(obs.open_tickets_summary)}\n"
    prompt += f"Last Action Status: {obs.last_action_status}\n"
    
    if obs.currently_viewed_ticket:
        prompt += f"\nCurrently Viewed Ticket Details:\n"
        prompt += f"{obs.currently_viewed_ticket.model_dump_json(indent=2)}\n"
        
    return prompt.strip()

def run_task(client: OpenAI, task_name: str, model_name: str = MODEL_NAME) -> Tuple[float, List[str]]:
    trace = []
    print(f"\n{'='*40}\nRunning Task: {task_name.upper()}\n{'='*40}")
    env = CustomerSupportEnv(task_name=task_name)
    obs = env.reset()
    
    history = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]
    
    for step in range(1, MAX_STEPS + 1):
        response_text = ""
        if env.is_done:
            break
            
        user_prompt = build_user_prompt(step, obs)
        history.append({"role": "user", "content": user_prompt})
        
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=history,
                temperature=0.0
            )
            response_text = completion.choices[0].message.content.strip()
            
            # Robust JSON Extraction
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1:
                clean_json = response_text[start:end+1]
            else:
                clean_json = response_text

            action_data = json.loads(clean_json)
            action = Action(**action_data)
        except Exception as e:
            msg = f"Step {step} Error: {e}"
            print(f"[{step}] Model error or invalid JSON: {e}")
            response_text = f"Error: {e}"
            action = Action(action_type="submit") # abort
            trace.append(msg)

        step_log = f"Step {step}: {action.action_type}"
        if action.ticket_id: step_log += f" ({action.ticket_id})"
        print(f"[{step}] Model Action: {action.model_dump_json()}")
        history.append({"role": "assistant", "content": response_text})
        
        obs, reward, done, info = env.step(action)
        step_log += f" -> {obs.last_action_status[:50]}... [Reward: {reward.value}]"
        trace.append(step_log)
        print(f"      Reward: {reward.value} ({reward.reason}) | Done: {done}")

    # Grade
    final_state = env.state()
    from models import Ticket
    final_tickets = [Ticket(**t) for t in final_state["tickets"]]
    score = env.task.grade(final_tickets)
    print(f"Task '{task_name}' completed. Final Grader Score: {score:.2f}")
    return score, trace

def run_benchmark(
    task_names: List[str],
    api_key: str,
    model_name: str = MODEL_NAME,
    api_base_url: str = API_BASE_URL
) -> Dict[str, Any]:
    """Runs selected tasks and returns grader scores and traces."""
    client = OpenAI(base_url=api_base_url, api_key=api_key)
    results: Dict[str, Any] = {}
    for task_name in task_names:
        score, trace = run_task(client, task_name, model_name=model_name)
        results[task_name] = {"score": score, "trace": trace}
    return results

def main():
    if not API_KEY or API_KEY == "dummy-key":
        print("Warning: No API_KEY set. Ensure OPENAI_API_KEY is defined in real usage.")
        
    scores = run_benchmark(
        task_names=["easy", "medium", "hard"],
        api_key=API_KEY,
        model_name=MODEL_NAME,
        api_base_url=API_BASE_URL
    )
        
    print("\n\n" + "-"*30)
    print("FINAL SCORES")
    print("-" * 30)
    for t, s in scores.items():
        print(f"{t.ljust(10)}: {s:.2f}/1.00")
        
if __name__ == "__main__":
    main()
