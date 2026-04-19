from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage  

from difflib import get_close_matches
from pathlib import Path

import json

MAX_RETRY = 2

_ = load_dotenv()
with open(Path(__file__).parent / "cities.json") as f:
    CITY_DB = json.load(f)

CITY_DB = {k.lower(): v.lower() for k, v in CITY_DB.items()}
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
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


class ExtractedInfo(BaseModel):
    city: Optional[str] = Field(default=None, description="City name")
    country: Optional[str] = Field(default=None, description="Country name")
    intent: Optional[str] = Field(default=None, description="Intent of the query")

# print(cities)
structured_llm = llm.with_structured_output(ExtractedInfo)

def extractor_node(state: SanitizerState) -> SanitizerState:
    try:
        response = structured_llm.invoke(prompt.format(query=state["raw_query"]))
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
    # print("1. Extracted: ", state)
    extracted = state.get("extracted")
    if not extracted:
        return {
            **state,
            "validated": False,
            "errors": ["NO_EXTRACTION_DATA"],
        }
    # extracted = extracted.model_dump()
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
    elif city_norm not in CITY_DB:
        errors.append("CITY_NOT_FOUND")
    
    # missing country 
    else:
        correct_country = CITY_DB.get(city_norm)
        if country_norm and country_norm!=correct_country:
            errors.append(f"COUNTRY_MISMATCH")
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
                "corrections": [f"Corrected city from {city} to {corrected_city}"],
                "validated": True,
                "errors": [],
            }
    return state
            
def validation_router(state: SanitizerState) -> str:
    if state["validated"]:
        return "end"
    if state["retry_count"] >= MAX_RETRY:
        return "end"
    return "corrector"

def increment_retry(state: SanitizerState) -> SanitizerState:
    if state["errors"]:
        return {**state, "retry_count": state["retry_count"] + 1}
    return state
# for v2
def llm_correction_node(state: SanitizerState):
    extracted = state.get("extracted")
    errors = state["errors"]

    system_prompt = """
    You are a correction AI. Your job is to fix invalid city names.
    If the extracted city is invalid, guess the correct one based on spelling,
    phonetics, and common world cities. 
    Only output a corrected city name. If unsure, output null.
    """
    user_prompt = f"""
    Extracted city: {extracted}
    Errors: {errors}
    Query: {state['raw_query']}
    Output ONLY the corrected city or null.
    """
    resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    cleaned = resp.content.strip().lower()
    if cleaned in ["null","none","", "undefined"]:
        return {**state, "llm_city_guess":  None}
    # print(f"resp.content {cleaned}")
    return {**state, "llm_city_guess":  cleaned}
def apply_llm_guess_node(state: SanitizerState):
    if state["validated"]:
        return state
    llm_guess = state.get("llm_city_guess")
    if not llm_guess:
        return state
    new_ext = {**state["extracted"], "city":llm_guess,"country":CITY_DB.get(llm_guess) }
    return {**state, "extracted": new_ext}
def llm_correction_router(state: SanitizerState) -> str:
    if state["validated"]:
        return "end"
    if state["retry_count"] >= MAX_RETRY:
        return "end"
    return "llm_correction"
    


def debug_wrapper(fn, name):
    def wrapper(state):
        print(f"\n--- {name} ---")
        print(state)
        return fn(state)
    return wrapper
        
builder = StateGraph(SanitizerState)

builder.add_node("extractor", debug_wrapper(extractor_node, "extractor"))
builder.add_node("validator", debug_wrapper(validator_node, "validator"))
builder.add_node("corrector", debug_wrapper(corrector_node, "corrector"))
# v2
builder.add_node("llm_correction", debug_wrapper(llm_correction_node, "llm_correction"))
builder.add_node("apply_llm_guess", debug_wrapper(apply_llm_guess_node, "apply_llm_guess"))

builder.set_entry_point("extractor")
# linear flow
builder.add_edge("extractor", "validator")
# conditional edge
builder.add_conditional_edges(
    "validator",
    validation_router,
    {
        "end": END,
        "corrector": "llm_correction",
    }
)
# v2
builder.add_conditional_edges(
    "apply_llm_guess",
    validation_router,
    {
        "end": END,
        "corrector": "llm_correction",
    }
)
builder.add_node("increment_retry",debug_wrapper(increment_retry, "increment_retry"))

# builder.add_edge("corrector", "increment_retry")
# v2
builder.add_edge("llm_correction", "increment_retry")
builder.add_edge("increment_retry", "validator")
    
    
initial_state = {
    "raw_query": "Tell me weather in Duabi",
    # "raw_query": "Tell me weather in loha",
    # "raw_query": "Tell me Tokyo tourism",
    "extracted": None,
    "validated": False,
    "errors": [],
    "corrections": [],
    "retry_count": 0
}

graph = builder.compile()

result = graph.invoke(initial_state)
print(result)

