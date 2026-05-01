import uuid
from langgraph.checkpoint.memory import MemorySaver
from agent.graph import create_agent

initial_state = {
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

if __name__ == "__main__":
    graph_memory = MemorySaver()
    graph = create_agent(checkpointer=graph_memory)

    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    print(f"--- Starting Agent Session: {session_id} ---")

    user_input = None
    while True:
        if user_input is None:
            result = graph.invoke(initial_state, config)
        else:
            graph.update_state(config, {"user_selection": user_input})
            result = graph.invoke(None, config)

        snapshot = graph.get_state(config)
        
        if snapshot.next and "process_hitl" in snapshot.next:
            print("\n[!] 🛑 Graph Paused! Human Intervention Required.")
            print(f"Reason: {snapshot.values.get('pending_issue')}")
            
            candidates = snapshot.values.get('hitl_candidates')
            if candidates:
                print(f"Did you mean one of these? {', '.join(candidates)}")
                
            user_input = input("\nEnter city name (or type 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting pipeline...")
                break
                
            print(f"\n--- Resuming graph with: '{user_input}' ---")
        else:
            print("\n✅ Graph Execution Complete.")
            print("Final Extracted Data:", snapshot.values.get("extracted"))
            break
