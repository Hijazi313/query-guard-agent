from agent.state import SanitizerState, ExtractedInfo
from agent.config import llm
from agent.prompts import extractor_prompt

llm_extractor = llm.with_structured_output(ExtractedInfo)

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
            }
        return {
            **state,
            "extracted": response.model_dump(),
            "errors": [],
            "confidence": response.confidence,  
            "source": "extraction",
            "status": "ok" if response.confidence >= 0.8 else "low_confidence",
            "score": response.confidence
        }
    except Exception as e:
        return {
            **state,
            "errors": [f"Extraction failed: {str(e)}"],
            "extracted": None,
        }
