from langchain_core.messages import AIMessage
from state import AgentState
from tools.inference_tools import query_finetuned_model_tool

def finalizer_node(state: AgentState):
    print("\n--- 🏁 FINALIZER AGENT STARTED ---")
    
    # 1. Get Context
    user_query = state.get("user_goal", "Tell me about your services.")
    model_url = state.get("model_url")
    
    if not model_url:
        return {
            "messages": [AIMessage(content="❌ Error: No model URL found. Did training complete?")],
            "status": "failed"
        }

    print(f"   🤖 Model: {model_url}")
    print(f"   ❓ Query: {user_query}")
    
    # 2. Call the Inference Tool
    try:
        response = query_finetuned_model_tool(user_query, model_url)
        
        print(f"   ✅ RESPONSE GENERATED!")
        print(f"   💬 Output: {response}")
        
        return {
            "messages": [AIMessage(content=f"Final Answer from Fine-Tuned Model:\n\n{response}")],
            "status": "completed",
            "final_response": response
        }
        
    except Exception as e:
        print(f"   ❌ Inference Failed: {e}")
        return {
            "messages": [AIMessage(content=f"Inference error: {str(e)}")],
            "error": str(e)
        }