from agent.state import SanitizerState, HITLDecision
from agent.config import log, MAX_RETRY, CITY_KEYS

def validation_router(state: SanitizerState) -> str:
    if state["validated"]:
        if state["confidence"] >= 0.85:
            return "end"
        else:
            log.warning("NEEDS_REVIEW", confidence=state["confidence"], source=state["source"], city=state["extracted"].get("city") if state.get("extracted") else None)
            return "end"
            
    if state.get("retry_count", 0) >= MAX_RETRY:
        return "llm_city_corrector_from_list"
        
    return "corrector"

def corrector_router(state: SanitizerState) -> str:
    if state["validated"]:
        return "validation"
        
    if state.get("retry_count", 0) >= MAX_RETRY:
        return "llm_city_corrector_from_list"
        
    if "CITY_NOT_FOUND" in state.get("errors",[]):
        return "llm_correction"
        
    return "end"

def hitl_router(state: SanitizerState) -> HITLDecision:
    if state.get("extracted") is None or not state["extracted"].get("city"):
        guess = state.get("llm_city_guess")
        candidates = [guess] if guess else []
        return HITLDecision(
            should_interrupt=True,
            reason="CITY_NOT_FOUND",
            candidates=candidates,
            resume_node="extractor",
        )
    
    if state.get("validated") == False and state.get("retry_count", 0) >= MAX_RETRY:
        guess = state.get("llm_city_guess")
        candidates = [guess] if guess else []
        return HITLDecision(
            should_interrupt=True,
            reason="EXHAUSTED_RETRIES",
            candidates=candidates,
            resume_node="extractor",
        )
        
    if state.get("confidence", 0.0) < 0.6:
        candidate = state.get("llm_city_guess")
        candidates = [candidate] if candidate else CITY_KEYS[:3]
        return HITLDecision(
            should_interrupt=True,
            reason="LOW_CONFIDENCE",
            candidates=candidates,
            resume_node="extractor",
        )
    
    return HITLDecision(
        should_interrupt=False,
        reason=None,
        candidates=None,
        resume_node=None
    )

def llm_city_corrector_from_list_router(state: SanitizerState) -> str:
    if hitl_router(state)["should_interrupt"]:
        return "hitl"
    return "end"
