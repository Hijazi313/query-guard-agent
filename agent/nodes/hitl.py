from agent.state import SanitizerState
from agent.routers import hitl_router
from agent.config import CITY_DB

def hitl_node(state: SanitizerState) -> SanitizerState:
    decision = hitl_router(state)
    return {
        **state,
        "awaiting_user": True,
        "pending_issue": decision["reason"],
        "hitl_candidates": decision["candidates"],
        "resume_node": decision["resume_node"]
    }

def process_hitl_node(state: SanitizerState) -> SanitizerState:
    selection = state.get("user_selection")
    if selection:
        selection_norm = selection.strip().lower()
        new_extracted = {
            "city": selection_norm,
            "country": CITY_DB.get(selection_norm),
            "intent": state.get("extracted", {}).get("intent") if state.get("extracted") else None,
            "confidence": 1.0,
            "source": "hitl"
        }
        return {
            **state,
            "extracted": new_extracted,
            "awaiting_user": False,
            "pending_issue": None,
            "hitl_candidates": None,
            "user_selection": None,
            "errors": [],
            "validated": False,
            "retry_count": 0
        }
    return state
