---
title: AI Enviroment
emoji: 🌍
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# Customer Support Ticket Triage Environment

A real-world OpenEnv-style environment for AI agent evaluation in customer support ticket triage. The agent must read, classify, assign priority, draft responses, and resolve or escalate tickets.

## Project Highlights
- Realistic support queue behavior with policy-based decisions.
- Reward shaping for incremental progress and penalties for invalid actions.
- Deterministic graders for `easy`, `medium`, and `hard` tasks with scores from `0.00` to `1.00`.
- Gradio app (`app.py`) for live Hugging Face Space demos.
- Scripted benchmark runner (`inference.py`) for local/CLI evaluation.

## Action Schema
The agent acts with the `Action` model in `models.py`.

Supported actions:
- `read_ticket`
- `classify_ticket`
- `assign_priority`
- `draft_response`
- `resolve_ticket`
- `escalate_ticket`
- `submit`

## Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set API key:
```bash
export OPENAI_API_KEY="sk-..."
```

3. Run CLI benchmark:
```bash
python inference.py
```

4. Run web demo:
```bash
python app.py
```
Then open `http://localhost:7860`.

## Hugging Face Space Deployment (Docker)

1. Create a new Hugging Face Space with **SDK = Docker**.
2. Push this repository to your Space remote.
3. In Space settings, add secret:
   - `OPENAI_API_KEY`
4. Wait for build to complete, then open the Space UI and run evaluations.

This repo already includes:
- `Dockerfile`
- `requirements.txt`
- `openenv.yaml`

## Submission Checklist (Hackathon)
- GitHub repo link: add here after push.
- Hugging Face Space link: add here after deployment.
- Space runs successfully and returns task scores.
