from agent.state import SanitizerState
from agent.config import CITY_DB

def validator_node(state: SanitizerState) -> SanitizerState:
    extracted = state.get("extracted")
    if not extracted:
        return {**state, "validated": False, "errors": ["NO_EXTRACTION_DATA"]}
    
    city = extracted.get("city")
    city_norm = city.lower() if city else None

    if city_norm in CITY_DB:
        return {
            **state,
            "validated": True,
            "confidence": 1.0,
            "source": "exact_match",
            "errors": []
        }
    
    return {**state, "validated": False, "errors": ["CITY_NOT_FOUND"]}
