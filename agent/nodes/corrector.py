from difflib import get_close_matches, SequenceMatcher
from agent.state import SanitizerState
from agent.config import CITY_DB, CITY_KEYS

def corrector_node(state: SanitizerState) -> SanitizerState:
    extracted = state.get("extracted")
    if not extracted:
        return state

    city_norm = extracted.get("city").lower() if extracted.get("city") else None
    
    if city_norm:
        matches = get_close_matches(city_norm, CITY_KEYS, n=1, cutoff=0.7)
        if matches:
            corrected_city = matches[0]
            score = SequenceMatcher(None, city_norm, corrected_city).ratio()
            
            new_extracted = {
                **extracted,
                "city": corrected_city,
                "country": CITY_DB[corrected_city]
            }
            return {
                **state,
                "extracted": new_extracted,
                "confidence": score,
                "source": "fuzzy",
                "corrections": state.get("corrections", []) + [f"fuzzy:{city_norm}->{corrected_city} ({score:.2f})"],
                "errors": [],
                "validated": True 
            }
    return state
