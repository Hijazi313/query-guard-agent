from langgraph.graph import StateGraph, END
from agent.state import SanitizerState
from agent.nodes import (
    extractor_node,
    validator_node,
    corrector_node,
    llm_correction_node,
    llm_city_corrector_from_list_node,
    hitl_node,
    process_hitl_node,
    increment_retry_node,
    debug_wrapper
)
from agent.routers import (
    validation_router,
    corrector_router,
    hitl_router,
    llm_city_corrector_from_list_router
)

builder = StateGraph(SanitizerState)

builder.add_node("extractor", debug_wrapper(extractor_node, "extractor"))
builder.add_node("validator", debug_wrapper(validator_node, "validator"))
builder.add_node("corrector", debug_wrapper(corrector_node, "corrector"))
builder.add_node("llm_correction", debug_wrapper(llm_correction_node, "llm_correction"))
builder.add_node("increment_retry", debug_wrapper(increment_retry_node, "increment_retry"))
builder.add_node("llm_city_corrector_from_list", debug_wrapper(llm_city_corrector_from_list_node, "llm_city_corrector_from_list"))
builder.add_node("hitl", debug_wrapper(hitl_node, "hitl"))
builder.add_node("process_hitl", debug_wrapper(process_hitl_node, "process_hitl"))

builder.set_entry_point("extractor")

builder.add_conditional_edges(
    "extractor", 
    lambda state: "hitl" if hitl_router(state)["should_interrupt"] else "validator",
    {
        "hitl": "hitl",
        "validator": "validator",
    }   
)

builder.add_edge("hitl", "process_hitl")
builder.add_edge("process_hitl", "validator")

builder.add_edge("llm_correction", "increment_retry")
builder.add_edge("increment_retry", "validator")

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

def create_agent(checkpointer=None):
    if checkpointer:
        return builder.compile(checkpointer=checkpointer, interrupt_before=["process_hitl"])
    return builder.compile()
