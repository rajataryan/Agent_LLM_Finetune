import os
import modal
from state import AgentState
# Import the tools that we just proved are working
from tools.training_tools import train_love_bot, app

# Enable output so we can see logs in the Streamlit terminal
modal.enable_output()

def training_node(state: AgentState):
    """Dispatches training job to Modal's serverless GPU."""
    print("\n" + "="*60)
    print("--- 🚀 TRAINING AGENT STARTED ---")
    print("="*60)
    
    dataset_file = state.get("training_file_path")
    project_name = state.get("project_name", "love-bot")
    
    # Sanitize project name for Hugging Face (lowercase, no spaces)
    safe_project_name = project_name.strip().replace(" ", "-").replace("_", "-").lower()
    
    # 1. Validation
    if not dataset_file:
        print("   ❌ ERROR: No dataset file path provided")
        return {"training_status": "failed", "error": "No dataset file found"}
        
    if not os.path.exists(dataset_file):
        print(f"   ❌ ERROR: File not found: {dataset_file}")
        return {"training_status": "failed", "error": f"File not found: {dataset_file}"}

    # 2. Read Data
    print(f"   📂 Reading: {dataset_file}")
    try:
        with open(dataset_file, "rb") as f:
            dataset_bytes = f.read()
        print(f"   ✅ Loaded: {len(dataset_bytes)} bytes")
    except Exception as e:
        return {"training_status": "failed", "error": f"Read error: {e}"}
        
    # 3. Dispatch to Cloud
    print(f"   🚀 Dispatching to Modal: '{safe_project_name}'")
    print("   ⏳ This will take 3-5 minutes (GPU warm-up + training)...")
    
    try:
        # We use the context manager because this is how we ran the successful test
        with app.run():
            model_url = train_love_bot.remote(
                dataset_bytes=dataset_bytes, 
                project_name=safe_project_name
            )
            
        print(f"   ✅ SUCCESS! Model: {model_url}")
        print("="*60 + "\n")
        
        return {
            "training_status": "success", 
            "model_url": model_url,
            "messages": [f"✅ Training complete! Model: {model_url}"]
        }
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        print("="*60 + "\n")
        return {"training_status": "failed", "error": str(e)}