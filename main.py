import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from state import AgentState

# Import ALL Agents
from agents.intake_agent import intake_node
from agents.browser_agent import browser_node
from agents.data_agent import data_node
from agents.training_agent import training_node
from agents.finalizer_agent import finalizer_node 

load_dotenv(override=True)

# --- ROUTING LOGIC ---
def check_browser_status(state: AgentState):
    content = state.get("dataset_content", "")
    if not content:
        print("   🔄 Loop: No content. Retrying Browser...")
        return "browser_agent"
    return "data_agent"

def check_data_progress(state: AgentState):
    current = state.get("generated_count", 0)
    target = state.get("dataset_size", 10)
    if current < target:
        print(f"   🔄 Loop: {current}/{target} samples. Generating more...")
        return "data_agent"
    print("   ✅ Data ready. Moving to TRAINING.")
    return "training_agent"

def check_training_status(state: AgentState):
    status = state.get("training_status")
    if status == "success":
        print("   ✅ Training done. Moving to INFERENCE.")
        return "finalizer_agent"
    print("   ❌ Training failed. Stopping.")
    return END

# --- GRAPH BUILD ---
def build_graph():
    builder = StateGraph(AgentState)

    # Add Nodes
    builder.add_node("intake_agent", intake_node)
    builder.add_node("browser_agent", browser_node)
    builder.add_node("data_agent", data_node)
    builder.add_node("training_agent", training_node)
    builder.add_node("finalizer_agent", finalizer_node) # <--- NEW NODE

    # Add Edges
    builder.add_edge(START, "intake_agent")
    builder.add_edge("intake_agent", "browser_agent")
    
    builder.add_conditional_edges("browser_agent", check_browser_status, 
                                  {"browser_agent": "browser_agent", "data_agent": "data_agent"})
    
    builder.add_conditional_edges("data_agent", check_data_progress, 
                                  {"data_agent": "data_agent", "training_agent": "training_agent"})
    
    # New Edge: Training -> Finalizer
    builder.add_conditional_edges("training_agent", check_training_status,
                                  {"finalizer_agent": "finalizer_agent", END: END})
    
    builder.add_edge("finalizer_agent", END)

    return builder.compile()

if __name__ == "__main__":
    print("🚀 BOOTING UP AI FACTORY...")
    graph = build_graph()
    
    # --- TEST RUN ---
    # We pretend to be a user asking for a Sofa Support Bot
    initial_state = {
        "messages": [],
        "user_goal": "I want a bot that answers questions about ComfyCloud Sofas return policy.", 
    }
    
    events = graph.stream(initial_state, config={"recursion_limit": 100})
    for event in events:
        pass # The agents define their own prints
        
    print("\n🏁 PIPELINE FINISHED.")