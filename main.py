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

from difflib import get_close_matches
from pathlib import Path

import json

MAX_RETRY = 2

with open(Path(__file__).parent / "cities.json") as f:
    CITY_DB = json.load(f)

CITY_DB = {k.lower(): v.lower() for k, v in CITY_DB.items()}
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

class LLMCorrection(BaseModel):
    city: Optional[str]

class ExtractedInfo(BaseModel):
    city: Optional[str] = Field(default=None, description="City name")
    country: Optional[str] = Field(default=None, description="Country name")
    intent: Optional[str] = Field(default=None, description="Intent of the query")

llm_extractor = llm.with_structured_output(ExtractedInfo)
llm_corrector = llm.with_structured_output(LLMCorrection)

def extractor_node(state: SanitizerState) -> SanitizerState:
    try:
        response = llm_extractor.invoke(extractor_prompt.format_messages(query=state["raw_query"]))
        return {
            **state,
            "extracted": response.model_dump(),
            "errors": [],
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
    errors = []
    if not extracted:
        return {
            **state,
            "validated": False,
            "errors": ["NO_EXTRACTION_DATA"],
        }
    city = extracted.get("city")
    country = extracted.get("country")
    # intent = extracted.get("intent")
    # normalize
    city_norm = city.lower() if city else None
    country_norm = country.lower() if country else None

    # missing city 
    if not city_norm:
        errors.append("CITY_MISSING")
        return {**state, "errors": errors, "validated": False}
    elif city_norm not in CITY_DB:
        errors.append("CITY_NOT_FOUND")
        return {**state, "errors": errors, "validated": False}
    
    # missing country 
    correct_country = CITY_DB.get(city_norm)
    if country_norm and country_norm!=correct_country:
        errors.append(f"COUNTRY_MISMATCH")
        # return {**state, "errors": errors, "validated": False}
    return {**state,
            "errors": errors,
            "validated": len(errors) == 0,
            }
    
def corrector_node(state: SanitizerState) -> SanitizerState:
    extracted = state.get("extracted")
    errors = state.get("errors", [])
    if not extracted:
        return state
    city = extracted.get("city")
    city_norm = city.lower() if city else None
    # only try to fix city not found
    if "CITY_NOT_FOUND" in errors and city_norm:
        matches = get_close_matches(word=city_norm, possibilities=list(CITY_DB.keys()), n=1, cutoff=0.7)
        if matches:
            corrected_city = matches[0]
            corrected_country = CITY_DB[corrected_city]
            # Apply correction into state
            new_extracted = {
                **extracted,
                "city": corrected_city,
                "country": corrected_country
            }
            return {
                **state,
                "extracted": new_extracted,
                # "corrections": [f"Corrected city from {city} to {corrected_city}"],
                 "corrections": state["corrections"] + [f"fuzzy:{city}->{corrected_city}"],

                # "validated": True,
                "retry_count":state["retry_count"]+1,
                "errors": [],
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
    - fix typos
    - find closest match
    - map to real cities
    """

    user_prompt = f"""
    Extracted: {extracted}
    Query: {state['raw_query']}
    """

    try:
        response = llm_corrector.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
       
        return {
            **state,
            "llm_city_guess":response.city.lower() if response.city else None,
        }
    except Exception:
        return {
            **state,
            "llm_city_guess": None,
            # "errors": state["errors"] + ["LLM_CORRECTION_FAILED"]
        }
    
def apply_llm_guess_node(state: SanitizerState):
    llm_guess = state.get("llm_city_guess")

    if not llm_guess or llm_guess not in CITY_DB:
        return state
    
    new_ext = {**state["extracted"], "city":llm_guess,"country":CITY_DB.get(llm_guess) }
    return {**state, "extracted": new_ext, "errors":[],
     "corrections": state["corrections"] + [f"llm:{llm_guess}"],}




# Conditional Routers
def validation_router(state: SanitizerState) -> str:
    if state["validated"]:
        return "end"
    if state["retry_count"] >= MAX_RETRY:
        return "end"
    return "corrector"

def corrector_router(state: SanitizerState) -> str:
    if state["validated"]:
        return "end"
    if state["retry_count"] >= MAX_RETRY:
        return "end"
    if "CITY_NOT_FOUND" in state.get("errors",[]):
        return "llm_correction"
    return "end"
    # return "validator"  

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
builder.add_node("apply_llm_guess", debug_wrapper(apply_llm_guess_node, "apply_llm_guess"))
builder.add_node("increment_retry",debug_wrapper(increment_retry, "increment_retry"))

builder.set_entry_point("extractor")
# # linear flow
builder.add_edge("extractor", "validator")
builder.add_edge("llm_correction", "apply_llm_guess")
builder.add_edge("apply_llm_guess", "increment_retry")
builder.add_edge("increment_retry", "validator")

# # conditional edge
builder.add_conditional_edges(
    "validator",
    validation_router,
    {
        "end": END,
        "corrector": "corrector",
    }
)

builder.add_conditional_edges(
    "corrector",
    corrector_router,
    {
        "llm_correction": "llm_correction",
        "end": END,
    }
)



    
    
initial_state = {
    "raw_query": "Tell me weather in Duabioi",
    "extracted": None,
    "validated": False,
    "errors": [],
    "corrections": [],
    "retry_count": 0,
    "llm_city_guess": None, 
}

graph = builder.compile()

result = graph.invoke(initial_state)
log.info(result)

