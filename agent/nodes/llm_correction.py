from langchain_core.messages import HumanMessage, SystemMessage
from difflib import get_close_matches
from agent.state import SanitizerState, LLMCorrection
from agent.config import llm, CITY_DB, CITY_KEYS

llm_corrector = llm.with_structured_output(LLMCorrection)

def llm_correction_node(state: SanitizerState) -> SanitizerState:
    extracted = state.get("extracted")
    if not extracted:
        return state

    system_prompt = """
    You correct invalid city names. Return ONLY valid JSON:
    { "city": "correct_city_or_null" }

    Rules:
    - If unsure → return null
    - No extra text
    - map to real cities
    """
    user_prompt = f"Extracted: {extracted}\nQuery: {state['raw_query']}"

    try:
        response = llm_corrector.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        guess = response.city.lower() if response.city else None
        
        if guess and guess in CITY_DB:
            return {
                **state,
                "llm_city_guess": guess,
                "extracted": {**state["extracted"], "city": guess, "country": CITY_DB[guess]},
                "confidence": 0.65,
                "source": "llm",
                "validated": True,
                "corrections": state.get("corrections", []) + [f"llm:{guess}"]
            }
        return {**state, "llm_city_guess": guess}
    except Exception:
        return state

def llm_city_corrector_from_list_node(state: SanitizerState) -> SanitizerState:
    extracted = state.get("extracted")
    if not extracted or not extracted.get("city"):
        return state
    
    candidates = get_close_matches(extracted.get("city"), CITY_KEYS, n=10, cutoff=0.3)

    system_prompt = f"""You are a city guessing engine.
    The user has mistyped or misspelled a city name.
    Your job: guess the correct city from the list of valid cities provided.

    Valid cities:
    {candidates}

    Return ONLY valid JSON:
    {{ "city": "correct_city_or_null" }}

    Rules:
    - If unsure → return null
    - No extra text
    - Map to real cities from the list above
    """

    user_prompt = f"Extracted: {extracted}\nQuery: {state['raw_query']}"

    try:
        response = llm_corrector.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        guess = response.city.lower() if response.city else None
        
        if guess and guess in CITY_DB:
            return {
                **state,
                "llm_city_guess": guess,
                "extracted": {**state["extracted"], "city": guess, "country": CITY_DB[guess]},
                "confidence": 0.60,
                "source": "llm_list",
                "validated": True,
                "corrections": state.get("corrections", []) + [f"llm_list:{guess}"]
            }
        return state
    except Exception:
        return state
