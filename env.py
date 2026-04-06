import copy
from typing import Tuple, Dict, Any, Optional

from models import Action, Observation, Reward, Ticket
from tasks import get_task, BaseTask

class CustomerSupportEnv:
    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self.task: BaseTask = get_task(task_name)
        self.tickets = []
        self.current_viewed_ticket_id = None
        self.last_action_status = ""
        self.is_done = False

    def reset(self) -> Observation:
        self.tickets = self.task.reset_tickets()
        self.current_viewed_ticket_id = None
        self.last_action_status = "Environment initialized. Awaiting action."
        self.is_done = False
        return self._make_observation()

    def state(self) -> Dict[str, Any]:
        """Returns internal ground truth state, used by Grader."""
        return {
            "task_name": self.task.name,
            "tickets": [t.model_dump() for t in self.tickets],
            "done": self.is_done
        }

    def _make_observation(self) -> Observation:
        open_tickets = [f"{t.id}" for t in self.tickets if t.status not in ["resolved", "escalated"]]
        
        currently_viewed = None
        if self.current_viewed_ticket_id:
            for t in self.tickets:
                if t.id == self.current_viewed_ticket_id:
                    currently_viewed = t
                    break

        return Observation(
            goal=self.task.goal,
            open_tickets_summary=open_tickets,
            last_action_status=self.last_action_status,
            currently_viewed_ticket=currently_viewed
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.last_action_status = ""
        reward_value = 0.0
        reward_reason = ""
        
        # Verify if an action requires a ticket ID
        req_ticket_actions = [
            "read_ticket", "classify_ticket", "assign_priority", 
            "draft_response", "resolve_ticket", "escalate_ticket"
        ]

        if action.action_type == "submit":
            self.last_action_status = "Submit evaluated. Episode done."
            self.is_done = True
            reward_value = 0.0
            reward_reason = "Episode terminated."
            return self._make_observation(), Reward(value=reward_value, reason=reward_reason), self.is_done, {}

        if action.action_type in req_ticket_actions and not action.ticket_id:
            self.last_action_status = f"Error: {action.action_type} requires a ticket_id."
            reward_value = -0.1
            reward_reason = "Invalid action format."
            return self._make_observation(), Reward(value=reward_value, reason=reward_reason), self.is_done, {}

        target_ticket = next((t for t in self.tickets if t.id == action.ticket_id), None)
        
        if action.action_type in req_ticket_actions and not target_ticket:
            self.last_action_status = f"Error: Ticket {action.ticket_id} not found."
            reward_value = -0.1
            reward_reason = "Target ticket does not exist."
            return self._make_observation(), Reward(value=reward_value, reason=reward_reason), self.is_done, {}

        if action.action_type == "read_ticket":
            self.current_viewed_ticket_id = action.ticket_id
            self.last_action_status = f"Now viewing ticket {action.ticket_id}."
            reward_value = 0.1
            reward_reason = "Successfully read a ticket."

        elif action.action_type == "classify_ticket":
            if action.issue_type:
                target_ticket.issue_type = action.issue_type
                self.last_action_status = f"Ticket {action.ticket_id} classified as {action.issue_type}."
                reward_value = 0.1
                reward_reason = "Classified ticket."
            else:
                self.last_action_status = "Error: classify_ticket requires issue_type."
                reward_value = -0.1
                reward_reason = "Missing fields."

        elif action.action_type == "assign_priority":
            if action.priority:
                target_ticket.priority = action.priority
                self.last_action_status = f"Ticket {action.ticket_id} priority set to {action.priority}."
                reward_value = 0.1
                reward_reason = "Assigned priority."
            else:
                self.last_action_status = "Error: assign_priority requires priority."
                reward_value = -0.1
                reward_reason = "Missing fields."

        elif action.action_type == "draft_response":
            if action.response_text:
                target_ticket.draft_response = action.response_text
                self.last_action_status = f"Response drafted for {action.ticket_id}."
                reward_value = 0.1
                reward_reason = "Drafted a response."
            else:
                self.last_action_status = "Error: draft_response requires response_text."
                reward_value = -0.1
                reward_reason = "Missing fields."

        elif action.action_type == "resolve_ticket":
            target_ticket.status = "resolved"
            self.last_action_status = f"Ticket {action.ticket_id} resolved."
            self.current_viewed_ticket_id = None
            reward_value = 0.2
            reward_reason = "Resolved ticket."

        elif action.action_type == "escalate_ticket":
            target_ticket.status = "escalated"
            self.last_action_status = f"Ticket {action.ticket_id} escalated."
            self.current_viewed_ticket_id = None
            reward_value = 0.2
            reward_reason = "Escalated ticket."

        else:
            self.last_action_status = f"Unknown action: {action.action_type}"
            reward_value = -0.1
            reward_reason = "Unknown action."

        # Auto-complete if no open tickets
        open_count = sum(1 for t in self.tickets if t.status == "open")
        if open_count == 0:
            self.is_done = True
            
        return self._make_observation(), Reward(value=reward_value, reason=reward_reason), self.is_done, {}
