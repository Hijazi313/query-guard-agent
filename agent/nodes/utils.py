from typing import Optional, List
from agent.state import SanitizerState
from agent.config import log

def build_hitl_message(reason: str, candidates: Optional[List[str]]) -> str:
    if reason == "CITY_NOT_FOUND":
        return (
            "I couldn't find the city in your query. "
            "Did you mean one of these?"
        )
    elif reason == "LOW_CONFIDENCE":
        return (
            "I detected a city but I'm not confident. "
            "Did you mean one of these?"
        )
    elif reason == "EXHAUSTED_RETRIES":
        return (
            "I'm having trouble understanding the city in your query. "
            "Did you mean one of these?"
        )
    return "Please confirm your intended city."

def debug_wrapper(fn, name):
    def wrapper(state):
        log.info(f"--- {name} ---")
        log.info(state)
        return fn(state)
    return wrapper

def increment_retry_node(state: SanitizerState) -> SanitizerState:
    if state.get("errors"):
        return {**state, "retry_count": state.get("retry_count", 0) + 1}
    return state
