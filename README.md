# Customer Support Ticket Triage Environment

A complete, real-world OpenEnv environment for AI agents designed to simulate internal customer support queues. The agent must read, classify, assign priority, draft responses, and resolve or escalate tickets accurately.

## 🏁 Environment Features
- **Real-World Application**: Resolving tickets involves context-dependent policy application rather than toy tasks.
- **Progressive Reward Shaping**: Rewards partial logic (+0.1 for reading, +0.1 for classifying/priority assignment correctly) and penalizes hallucinations/looping (-0.1).
- **Difficulty Scaling**: Ships with deterministic Graders checking Easy, Medium, and Hard tasks mapping from 0.0 to 1.0 logic scores.
- **Baseline included**: Runs `inference.py` leveraging OpenAI LLMs sequentially over the step loop.

### Actions Space
AI agents interface with JSON output based on the `Action` model defined in `models.py`.
Possible actions: `read_ticket`, `classify_ticket`, `assign_priority`, `draft_response`, `resolve_ticket`, `escalate_ticket`, `submit`.

### Execution
Provide `OPENAI_API_KEY` to run the baseline evaluation zero-shot loop.
```bash
export OPENAI_API_KEY="sk-..."
python inference.py
```

### Hugging Face Deployment
Since this environment is Dockerized and registered with `openenv.yaml`:
1. Push this directory to a Hugging Face Space (Docker Template).
2. Configure Secrets (`OPENAI_API_KEY`) within HF settings.
3. OpenEnv framework tests via `openenv validate` automatically map to the `.yaml` definition.
