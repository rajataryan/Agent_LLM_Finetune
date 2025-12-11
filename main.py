import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from state import AgentState

# --- IMPORT AGENTS ---
from agents.intake_agent import intake_node
from agents.browser_agent import browser_node
from agents.data_agent import data_node
from agents.training_agent import training_node  # <--- The new Cloud Agent

load_dotenv(override=True)

# --- CONDITIONAL LOGIC ---

def check_browser_status(state: AgentState):
    """
    Checks if the Browser Agent successfully scraped content.
    If empty, loop back to try again (or let the agent handle fallback).
    """
    content = state.get("dataset_content", "")
    if not content:
        print("   🔄 Loop: Search/Scrape yielded no data. Retrying Browser Agent...")
        return "browser_agent"
    
    # If we have data, move to Data Agent
    return "data_agent"

def check_data_progress(state: AgentState):
    """
    Checks if we have enough training examples.
    If current count < target, loop back to Data Agent to generate more.
    If target met, move to Training Agent.
    """
    current = state.get("generated_count", 0)
    target = state.get("dataset_size", 50)
    
    if current < target:
        print(f"   🔄 Loop: Generated {current}/{target}. Requesting more data...")
        return "data_agent"
    
    print(f"   ✅ Data Target Met ({current}/{target}). Proceeding to TRAINING...")
    return "training_agent"

# --- BUILD THE GRAPH ---

def build_graph():
    builder = StateGraph(AgentState)

    # 1. Add Nodes
    builder.add_node("intake_agent", intake_node)
    builder.add_node("browser_agent", browser_node)
    builder.add_node("data_agent", data_node)
    builder.add_node("training_agent", training_node) # Node 4: The Factory

    # 2. Add Edges
    # Start -> Intake -> Browser
    builder.add_edge(START, "intake_agent")
    builder.add_edge("intake_agent", "browser_agent")
    
    # Browser Loop (Loop until content found)
    builder.add_conditional_edges(
        "browser_agent",
        check_browser_status,
        {
            "browser_agent": "browser_agent", 
            "data_agent": "data_agent"
        }
    )
    
    # Data Loop (Loop until dataset size met)
    # If done -> Go to Training Agent
    builder.add_conditional_edges(
        "data_agent",
        check_data_progress,
        {
            "data_agent": "data_agent", 
            "training_agent": "training_agent"
        }
    )
    
    # Training -> End
    builder.add_edge("training_agent", END)

    return builder.compile()

# --- RUN BLOCK ---
if __name__ == "__main__":
    print("🚀 BOOTING UP AI PIPELINE...")
    
    graph = build_graph()
    
    # Test Input
    initial_state = {
        "messages": [],
        "user_goal": "Create a chatbot that speaks in Gen Z slang", 
    }
    
    print("--- STARTING EXECUTION ---")
    
    # Recursion limit must be high to allow for multiple data generation loops
    events = graph.stream(initial_state, config={"recursion_limit": 150})
    
    for event in events:
        for key, value in event.items():
            print(f"\n✅ FINISHED NODE: {key}")
            
            # Print helpful status updates based on context
            if "generated_count" in value:
                print(f"   📊 Dataset Progress: {value['generated_count']} samples.")
            
            if "model_url" in value:
                print(f"   🎉 SUCCESS! Model URL: {value['model_url']}")
                
    print("\n🏁 PIPELINE FINISHED.")