from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Ticket(BaseModel):
    id: str
    text: str
    issue_type: Optional[str] = None
    priority: Optional[str] = None
    status: Literal["open", "resolved", "escalated"] = "open"
    draft_response: Optional[str] = None

class Action(BaseModel):
    """
    Typed Action Model for the OpenEnv Agent. 
    The AI agent outputs this JSON format to take steps in the environment.
    """
    action_type: Literal[
        "read_ticket", 
        "classify_ticket", 
        "assign_priority", 
        "draft_response", 
        "resolve_ticket", 
        "escalate_ticket",
        "submit"
    ] = Field(description="The action taking place.")
    ticket_id: Optional[str] = Field(None, description="The ID of the ticket targeted by this action.")
    issue_type: Optional[str] = Field(None, description="Used only for classify_ticket action. E.g., 'billing', 'technical', 'inquiry'.")
    priority: Optional[str] = Field(None, description="Used only for assign_priority action. E.g., 'low', 'medium', 'high'.")
    response_text: Optional[str] = Field(None, description="Used only for draft_response action.")

class Observation(BaseModel):
    """
    Typed Observation Model for the OpenEnv State. 
    The Agent observes this after each step.
    """
    goal: str = Field(description="The overarching instruction for the episode.")
    open_tickets_summary: List[str] = Field(description="Summary of currently open tickets, e.g., 'Ticket 123'.")
    last_action_status: str = Field(description="Success or error message of the previous action.")
    currently_viewed_ticket: Optional[Ticket] = Field(None, description="The ticket specifically pulled up by read_ticket.")

class Reward(BaseModel):
    """
    Typed Reward Model for OpenEnv.
    """
    value: float = Field(description="The reward float value given at this step.")
    reason: str = Field(description="The explanation of why this reward was allocated.")
