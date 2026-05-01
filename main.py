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

from typing import TypedDict, Optional, List, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage  
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import Interrupt

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
- confidence: score between 0.0 and 1.0

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
    awaiting_user: bool = False # Flag to indicate if the graph is waiting for user input.
    pending_issue: Optional[str] = None  # Why HITL was triggered.  Foe example "LOW_CONFIDENCE" | "CITY_NOT_FOUND" | "EXHAUSTED_RETRIES"
    hitl_candidates: Optional[List[str]] = None # List of suggested city options shown to the user.
    user_selection: Optional[str] = None # City selected by the user when resuming.
    # resume_node: Optional[str] = None # The node where the graph must continue after the user replies.
    resume_node: Optional[Literal["extractor", "validator", "corrector", "llm_correction", "llm_city_corrector_from_list"]] = None # The node where the graph must continue after the user replies.
    status: Optional[str]  # "ok" | "low_confidence" | "no_city"
    score: float

class HITLDecision(TypedDict):
    should_interrupt: bool
    reason: Optional[str]
    candidates: Optional[List[str]]
    resume_node: Optional[str]
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

# External Functions
# Utils
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
    # fallback
    return "Please confirm your intended city."

def debug_wrapper(fn, name):
    def wrapper(state):
        log.info(f"--- {name} ---")
        log.info(state)
        return fn(state)
    return wrapper



# Nodes    
def extractor_node(state: SanitizerState) -> SanitizerState:
    try:

        response = llm_extractor.invoke(extractor_prompt.format_messages(query=state["raw_query"]))
        if not response.city:
            return {
                **state,
                "extracted": None,
                "errors": ["CITY_NOT_FOUND"],
                "confidence": 0.0,
                "source": "extraction",
                "status": "no_city",
                "score": 0.0,
                # "pending_issue": "NO_EXTRACTION_DATA",
                # "awaiting_user": True,
                # "hitl_candidates": [],
                # "extracted": None,
                # "resume_node": "extractor",
            }
        return {
            **state,
            "extracted": response.model_dump(),
            "errors": [],
            # "confidence": 0.0,  # Initialize
            "confidence": response.confidence,  
            "source": "extraction",
            "status": "ok" if response.confidence >= 0.8 else "low_confidence",
            "score": response.confidence
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
    

def hitl_node(state: SanitizerState):
    """
    Prepares the state so your UI/App knows why we paused.
    """
    decision = hitl_router(state)
    return {
        "awaiting_user": True,
        "pending_issue": decision["reason"],
        "hitl_candidates": decision["candidates"],
        "resume_node": decision["resume_node"]
    }

def process_hitl_node(state: SanitizerState):
    """
    Runs AFTER the user provides input to update the state.
    """
    selection = state.get("user_selection")
    if selection:
        selection_norm = selection.strip().lower()

        new_extracted = {
            "city": selection_norm,
            "country": CITY_DB.get(selection_norm),
            "intent": state.get("extracted", {}).get("intent") if state.get("extracted") else None,
            "confidence": 1.0,  # Human answered, so we are 100% confident
            "source": "hitl"
        }
        return {
            "extracted": new_extracted,
            "awaiting_user": False,
            "pending_issue": None,
            "hitl_candidates": None,
            "user_selection": None, # Clear the input
            "errors": [],
            "validated": False, # Send back to validator to be safe
            "retry_count": 0    # L6 Critique Fix: Reset retries so if human makes a typo, autocorrect runs again!
        }
    return state

def increment_retry_node(state: SanitizerState) -> SanitizerState:
    if state["errors"]:
        return {**state, "retry_count": state["retry_count"] + 1}
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
    # L6 Critique Fix: You must actually EVALUATE the hitl_router here!
    if hitl_router(state)["should_interrupt"]:
        return "hitl"
    return "end"
    
def hitl_router(state: SanitizerState) -> HITLDecision:
    # 1. Extractor returned no city at all
    if state["extracted"] is None or not state["extracted"].get("city"):
        guess = state.get("llm_city_guess")
        candidates = [guess] if guess else []
        return HITLDecision(
            should_interrupt=True,
            reason="CITY_NOT_FOUND",
            # candidates=[state["llm_city_guess"]],
            # candidates= state.get("llm_city_guess") and [state["llm_city_guess"]],
            candidates= candidates,
            # resume_node="corrector",
            resume_node="extractor",
        )
    if  state["validated"]== False and state["retry_count"] >= MAX_RETRY:
        return HITLDecision(
            should_interrupt=True,
            reason="EXHAUSTED_RETRIES",
            candidates= state.get("llm_city_guess") and [state["llm_city_guess"]],
            resume_node="extractor",
        )
    #    return interrupt("CITY_NOT_FOUND", candidates=[state["llm_city_guess"]], 
    #    )
     # 2. Extractor has a city but confidence too low
    if state["confidence"] < 0.6:
        # return interrupt("LOW_CONFIDENCE", candidates=[state["llm_city_guess"]])
        candidate= state.get("llm_city_guess")
        candidates = [candidate] if candidate else CITY_KEYS[:3] # send only 3 candidates to avoid overwhelming the user

        return HITLDecision(
            should_interrupt=True,
            reason="LOW_CONFIDENCE",
            candidates=candidates,
            resume_node="extractor",
        )
    
     # 3. Otherwise continue
    return {
        "should_interrupt": False,
        "reason": None,
        "candidates": None,
        "resume_node": None
    }



builder = StateGraph(SanitizerState)

builder.add_node("extractor", debug_wrapper(extractor_node, "extractor"))
builder.add_node("validator", debug_wrapper(validator_node, "validator"))
builder.add_node("corrector", debug_wrapper(corrector_node, "corrector"))
builder.add_node("llm_correction", debug_wrapper(llm_correction_node, "llm_correction"))
builder.add_node("increment_retry",debug_wrapper(increment_retry_node, "increment_retry"))
builder.add_node("llm_city_corrector_from_list",debug_wrapper(llm_city_corrector_from_list_node, "llm_city_corrector_from_list"))
builder.add_node("hitl", debug_wrapper(hitl_node, "hitl"))
builder.add_node("process_hitl", debug_wrapper(process_hitl_node, "process_hitl"))

builder.add_edge("hitl", "process_hitl")
builder.add_edge("process_hitl", "validator")

builder.set_entry_point("extractor")
# # linear flow
# builder.add_edge("extractor", "validator")
builder.add_conditional_edges(
    "extractor", 
    lambda state: "hitl" if hitl_router(state)["should_interrupt"] else "validator",
    {
        "hitl": "hitl",
        "validator": "validator",
    }   
)
builder.add_edge("llm_correction", "increment_retry")
builder.add_edge("increment_retry", "validator")

# # conditional edge
builder.add_conditional_edges(
    "validator",
    validation_router,
    {
        "hitl": "hitl",
        "end": END,
        "corrector": "corrector",
        "llm_city_corrector_from_list": "llm_city_corrector_from_list",
    }
)

builder.add_conditional_edges(
    "corrector",
    corrector_router,
    {
        "hitl": "hitl",
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
        "hitl": "hitl",
    }
)




    
    
initial_state = {
    # "raw_query": "Tell me weather in Duabioi",
    # "raw_query": "The capital of punjab Pakistan",
    # "raw_query": "weather in Duaawbiiii",
    "raw_query": "How's the weather",
    "extracted": None,
    "validated": False,
    "errors": [],
    "corrections": [],
    "retry_count": 0,
    "llm_city_guess": None, 
    "confidence": 0.0,
    "source": None,
    "awaiting_user": False,
    "pending_issue": None,
    "hitl_candidates": None,
    "user_selection": None,
    "resume_node": None,
}
import uuid

graph_memory = MemorySaver()
graph = builder.compile(checkpointer=graph_memory, interrupt_before=["process_hitl"])

# L6 Critique Fix: Generate a unique thread ID per run to avoid state bleeding across executions.
session_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": session_id}}

print(f"--- Starting Agent Session: {session_id} ---")

# L6 Critique Fix: A robust runner must be a loop. The graph might pause multiple times 
# if the user provides invalid input sequentially.
user_input = None
while True:
    if user_input is None:
        # Initial run
        result = graph.invoke(initial_state, config)
    else:
        # Resuming run with state injected
        graph.update_state(config, {"user_selection": user_input})
        result = graph.invoke(None, config)

    snapshot = graph.get_state(config)
    
    # Check if the graph is paused waiting for process_hitl
    if snapshot.next and "process_hitl" in snapshot.next:
        print("\n[!] 🛑 Graph Paused! Human Intervention Required.")
        print(f"Reason: {snapshot.values.get('pending_issue')}")
        
        candidates = snapshot.values.get('hitl_candidates')
        if candidates:
            print(f"Did you mean one of these? {', '.join(candidates)}")
            
        # Blocking terminal input
        user_input = input("\nEnter city name (or type 'quit' to exit): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting pipeline...")
            break
            
        print(f"\n--- Resuming graph with: '{user_input}' ---")
    else:
        # Graph reached the END
        print("\n✅ Graph Execution Complete.")
        print("Final Extracted Data:", snapshot.values.get("extracted"))
        break
