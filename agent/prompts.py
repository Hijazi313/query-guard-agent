from langchain_core.prompts import ChatPromptTemplate

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
