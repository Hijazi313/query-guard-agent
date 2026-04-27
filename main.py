import structlog
from dotenv import load_dotenv

_ = load_dotenv()

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()

from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage  

from difflib import get_close_matches, SequenceMatcher
from pathlib import Path

import json

MAX_RETRY = 2

with open(Path(__file__).parent / "cities.json") as f:
    CITY_DB = json.load(f)


CITY_DB = {k.lower(): v.lower() for k, v in CITY_DB.items()}
# Get all keys (city names)
CITY_KEYS = list(CITY_DB.keys())
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


    
extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an information extraction agent.

Extract the following fields from the user query:
- city
- country
- intent (e.g., weather, population, tourism)

Rules:
- Return NULL if not present
- Do NOT hallucinate
- Be strict and concise"""),
    ("human", "{query}")
])
class SanitizerState(TypedDict):
    raw_query: str
    extracted: Optional[dict]
    validated: bool
    errors: List[str]
    corrections: List[str]
    retry_count: int
    llm_city_guess: Optional[str] 
    confidence: float  # Added
    source: Optional[str]  # Added

class LLMCorrection(BaseModel):
    city: Optional[str]

class ExtractedInfo(BaseModel):
    city: Optional[str] = Field(default=None, description="City name")
    country: Optional[str] = Field(default=None, description="Country name")
    intent: Optional[str] = Field(default=None, description="Intent of the query")
    confidence: float = Field(default=0.0, description="Confidence score 0-1")
    source: Optional[str] = Field(default=None, description="Source of the extraction/correction")

llm_extractor = llm.with_structured_output(ExtractedInfo)
llm_corrector = llm.with_structured_output(LLMCorrection)

def extractor_node(state: SanitizerState) -> SanitizerState:
    try:
        response = llm_extractor.invoke(extractor_prompt.format_messages(query=state["raw_query"]))
        return {
            **state,
            "extracted": response.model_dump(),
            "errors": [],
            "confidence": 0.0,  # Initialize
            "source": "extraction"
        }


    except Exception as e:
        # Never let pipeline crash
        return {
            **state,
            "errors": [f"Extraction failed: {str(e)}"],
            "extracted": None,
        }

def validator_node(state: SanitizerState) -> SanitizerState:
    extracted = state.get("extracted")
    if not extracted:
        return {**state, "validated": False, "errors": ["NO_EXTRACTION_DATA"]}
    
    city = extracted.get("city")
    city_norm = city.lower() if city else None

    if city_norm in CITY_DB:
        # Exact Match Found
        return {
            **state,
            "validated": True,
            "confidence": 1.0,
            "source": "exact_match",
            "errors": []
        }
    
    return {**state, "validated": False, "errors": ["CITY_NOT_FOUND"]}
    
def corrector_node(state: SanitizerState) -> SanitizerState:
    extracted = state.get("extracted")
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
                "corrections": state["corrections"] + [f"fuzzy:{city_norm}->{corrected_city} ({score:.2f})"],
                "errors": [],
                "validated": True # The router will check confidence
            }
    return state
            


def llm_correction_node(state: SanitizerState):
    extracted = state.get("extracted")
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
                "corrections": state["corrections"] + [f"llm:{guess}"]
            }
        return {**state, "llm_city_guess": guess}
    except Exception:
        return state

# Load city anmes from the CITY_DB for  llm  and let him guess from the list
def llm_city_corrector_from_list_node(state: SanitizerState):
    extracted = state.get("extracted")
    
    # in order to avoid hitting context  limit , Send only top 10 candidates to LLM
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
                "confidence": 0.60, # Lower confidence for list-based guess
                "source": "llm_list",
                "validated": True,
                "corrections": state["corrections"] + [f"llm_list:{guess}"]
            }
        return state
    except Exception:
        return state
    
    

    

    




# Conditional Routers
def validation_router(state: SanitizerState) -> str:
    if state["validated"]:
        if state["confidence"] >= 0.85:
            return "end"
        else:
            log.warning("NEEDS_REVIEW", confidence=state["confidence"], source=state["source"], city=state["extracted"].get("city"))
            return "end" # Accept but flag in logs/traces
            
    if state["retry_count"] >= MAX_RETRY:
        return "llm_city_corrector_from_list"
        
    return "corrector"

def corrector_router(state: SanitizerState) -> str:
    if state["validated"]:
        return "validation" # Re-check confidence logic in validation_router
        
    if state["retry_count"] >= MAX_RETRY:
        return "llm_city_corrector_from_list"
        
    if "CITY_NOT_FOUND" in state.get("errors",[]):
        return "llm_correction"
        
    return "end"

def llm_city_corrector_from_list_router(state: SanitizerState) -> str:
    # After final fallback, we always end
    return "end"

# Utils
def debug_wrapper(fn, name):
    def wrapper(state):
        log.info(f"--- {name} ---")
        log.info(state)
        return fn(state)
    return wrapper
def increment_retry(state: SanitizerState) -> SanitizerState:
    if state["errors"]:
        return {**state, "retry_count": state["retry_count"] + 1}
    return state
builder = StateGraph(SanitizerState)

builder.add_node("extractor", debug_wrapper(extractor_node, "extractor"))
builder.add_node("validator", debug_wrapper(validator_node, "validator"))
builder.add_node("corrector", debug_wrapper(corrector_node, "corrector"))
builder.add_node("llm_correction", debug_wrapper(llm_correction_node, "llm_correction"))
builder.add_node("increment_retry",debug_wrapper(increment_retry, "increment_retry"))
builder.add_node("llm_city_corrector_from_list",debug_wrapper(llm_city_corrector_from_list_node, "llm_city_corrector_from_list"))

builder.set_entry_point("extractor")
# # linear flow
builder.add_edge("extractor", "validator")
builder.add_edge("llm_correction", "increment_retry")
builder.add_edge("increment_retry", "validator")

# # conditional edge
builder.add_conditional_edges(
    "validator",
    validation_router,
    {
        "end": END,
        "llm_city_corrector_from_list": "llm_city_corrector_from_list",
        "corrector": "corrector",
    }
)

builder.add_conditional_edges(
    "corrector",
    corrector_router,
    {
        "llm_correction": "llm_correction",
        "llm_city_corrector_from_list": "llm_city_corrector_from_list",
        "validation": "validator",
        "end": END,
    }
)
builder.add_conditional_edges(
    "llm_city_corrector_from_list",
    llm_city_corrector_from_list_router,
    {
        "end": END,
    }
)



    
    
initial_state = {
    # "raw_query": "Tell me weather in Duabioi",
    "raw_query": "The capital of punjab Pakistan",
    "extracted": None,
    "validated": False,
    "errors": [],
    "corrections": [],
    "retry_count": 0,
    "llm_city_guess": None, 
    "confidence": 0.0,
    "source": None
}

graph = builder.compile()

result = graph.invoke(initial_state)
log.info(result)

