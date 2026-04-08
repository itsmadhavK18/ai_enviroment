import json
from typing import List, Dict, Any
from models import Ticket

# Initial ticket definitions for the different tasks

EASY_TICKET = Ticket(
    id="T-001",
    text="Hi, I was double charged on my recent invoice statement. Can you look into this billing issue?",
)

MEDIUM_TICKETS = [
    Ticket(id="T-101", text="My dashboard is completely down, I keep getting a 500 server error when trying to deploy! Urgent!"),
    Ticket(id="T-102", text="How do I change my profile picture?"),
    Ticket(id="T-103", text="I lost my password but the reset link isn't reaching my email inbox."),
]

HARD_TICKETS = [
    Ticket(id="T-201", text="I want a refund! I bought this 45 days ago and it broke. I know policy is 30 days, but come on!"),
    Ticket(id="T-202", text="When is the new enterprise integration feature coming out?"),
    Ticket(id="T-203", text="Critical database latency in US-East-1. Requests timing out."),
    Ticket(id="T-204", text="I accidentally deleted my main project. Can you restore it from backup?"),
    Ticket(id="T-205", text="Cancel my subscription immediately."),
]

class BaseTask:
    def __init__(self, name: str, goal: str, initial_tickets: List[Ticket]):
        self.name = name
        self.goal = goal
        self.initial_tickets = [t.model_copy() for t in initial_tickets]

    def reset_tickets(self) -> List[Ticket]:
        return [t.model_copy() for t in self.initial_tickets]

    def grade(self, final_tickets: List[Ticket]) -> float:
        raise NotImplementedError

class EasyTask(BaseTask):
    def __init__(self):
        super().__init__(
            name="easy",
            goal="Classify the ticket as 'billing', draft a brief response acknowledging it, and resolve it.",
            initial_tickets=[EASY_TICKET]
        )

    def grade(self, final_tickets: List[Ticket]) -> float:
        if not final_tickets:
            return 0.0
        t = final_tickets[0]
        score = 0.0
        if t.issue_type == "billing":
            score += 0.3
        if t.draft_response and len(t.draft_response) > 5:
            score += 0.3
        if t.status == "resolved":
            score += 0.4
        return min(max(score, 0.0), 1.0)

class MediumTask(BaseTask):
    def __init__(self):
        super().__init__(
            name="medium",
            goal="Correctly triage 3 tickets. T-101 is a technical emergency (high priority -> escalate). T-102 is an inquiry (low priority -> resolve). T-103 is a technical issue (medium priority -> resolve).",
            initial_tickets=MEDIUM_TICKETS
        )

    def grade(self, final_tickets: List[Ticket]) -> float:
        score = 0.0
        tickets_by_id = {t.id: t for t in final_tickets}
        
        # T-101: Urgent technical
        t1 = tickets_by_id.get("T-101")
        if t1:
            if t1.issue_type == "technical": score += 0.1
            if t1.priority == "high": score += 0.1
            if t1.status == "escalated": score += 0.13

        # T-102: Simple inquiry
        t2 = tickets_by_id.get("T-102")
        if t2:
            if t2.issue_type == "inquiry": score += 0.1
            if t2.priority == "low": score += 0.1
            if t2.status == "resolved" and t2.draft_response: score += 0.13

        # T-103: Routine technical
        t3 = tickets_by_id.get("T-103")
        if t3:
            if t3.issue_type == "technical": score += 0.1
            if t3.priority == "medium": score += 0.1
            if t3.status == "resolved" and t3.draft_response: score += 0.14

        return min(max(score, 0.0), 1.0)

class HardTask(BaseTask):
    def __init__(self):
        super().__init__(
            name="hard",
            goal="""Triage 5 tickets with specific strict rules:
1. Out of policy refunds (T-201) must be priority 'low', resolved with a polite refusal.
2. Feature requests (T-202) -> issue_type 'inquiry', low priority, resolved.
3. System outages (T-203) -> issue_type 'technical', high priority, escalated.
4. Data loss/restores (T-204) -> issue_type 'technical', high priority, escalated.
5. Cancellations (T-205) -> issue_type 'billing', high priority, escalated.
""",
            initial_tickets=HARD_TICKETS
        )

    def grade(self, final_tickets: List[Ticket]) -> float:
        tickets_by_id = {t.id: t for t in final_tickets}
        score = 0.0
        weight_per_ticket = 1.0 / 5.0

        def grade_ticket(t, exp_type, exp_pri, exp_status):
            pts = 0.0
            if t and t.issue_type == exp_type: pts += 0.3 * weight_per_ticket
            if t and t.priority == exp_pri: pts += 0.3 * weight_per_ticket
            if t and t.status == exp_status: pts += 0.4 * weight_per_ticket
            return pts

        score += grade_ticket(tickets_by_id.get("T-201"), "billing", "low", "resolved")
        score += grade_ticket(tickets_by_id.get("T-202"), "inquiry", "low", "resolved")
        score += grade_ticket(tickets_by_id.get("T-203"), "technical", "high", "escalated")
        score += grade_ticket(tickets_by_id.get("T-204"), "technical", "high", "escalated")
        score += grade_ticket(tickets_by_id.get("T-205"), "billing", "high", "escalated")

        return min(max(score, 0.0), 1.0)

def get_task(task_name: str) -> BaseTask:
    if task_name == "easy":
        return EasyTask()
    elif task_name == "medium":
        return MediumTask()
    elif task_name == "hard":
        return HardTask()
    else:
        raise ValueError(f"Unknown task: {task_name}")
