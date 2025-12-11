import modal
import os
from langchain_core.messages import AIMessage
from state import AgentState
from tools.training_tools import train_model_on_modal, app 

def training_node(state: AgentState):
    print("--- 🚀 TRAINING AGENT STARTED ---")
    
    # 1. Validation
    file_path = state.get("training_file_path")
    if not file_path or not os.path.exists(file_path):
        return {
            "error": "Training file not found on local disk!", 
            "status": "failed",
            "messages": [AIMessage(content="Error: No training data found to send to the cloud.")]
        }
    
    # 2. Prepare Config
    with open(file_path, "r") as f:
        file_content = f.read()
        
    config = {
        "base_model": state.get("base_model", "unsloth/llama-3-8b-bnb-4bit"),
        "project_name": state.get("project_name", "my-finetune")
    }

    # 3. Trigger Modal
    print(f"   ⚡ Dispatching job to Modal (A10G GPU)...")
    print(f"   📂 Dataset size: {len(file_content)} bytes")
    print(f"   ⏳ If this is the first run, it may take 3-5 mins to build the image. Please wait...")
    
    try:
        # --- FIX: enable_output() MUST be the outer layer ---
        with modal.enable_output():
            with app.run():
                hf_url = train_model_on_modal.remote(file_content, config)
            
        print(f"   ✅ TRAINING COMPLETE!")
        print(f"   🔗 Model available at: {hf_url}")
        
        return {
            "training_status": "success",
            "model_url": hf_url,
            "status": "completed",
            "messages": [AIMessage(content=f"Training finished successfully! Your model is live at: {hf_url}")]
        }
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return {
            "training_status": "failed",
            "error": str(e),
            "status": "failed",
            "messages": [AIMessage(content=f"Training failed. Error: {str(e)}")]
        }