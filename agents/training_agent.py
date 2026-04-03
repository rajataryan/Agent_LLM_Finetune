import os
import modal
from state import AgentState
from tools.training_tools import train_generic_model, app

modal.enable_output()

def training_node(state: AgentState):
    print("\n" + "="*60)
    print("--- 🚀 TRAINING AGENT STARTED ---")
    print("="*60)
    
    dataset_file = state.get("training_file_path")
    project_name = state.get("project_name", "fine-tuned-model-v1")
    
    safe_project_name = project_name.strip().replace(" ", "-").replace("_", "-").lower()
    
    if not dataset_file or not os.path.exists(dataset_file):
        print("   ❌ ERROR: No dataset file found")
        return {"training_status": "failed", "error": "No dataset file found"}

    print(f"   📂 Reading: {dataset_file}")
    try:
        with open(dataset_file, "rb") as f:
            dataset_bytes = f.read()
        print(f"   ✅ Loaded: {len(dataset_bytes)} bytes")
    except Exception as e:
        return {"training_status": "failed", "error": f"Read error: {e}"}
        
    print(f"   🚀 Dispatching to Modal: '{safe_project_name}'")
    print("   ⏳ This will take 5-10 minutes.")
    print("   ⚠️  IMPORTANT: Do NOT let your Mac go to sleep or the connection will drop!")
    
    try:
        with app.run():
            model_url = train_generic_model.remote(
                dataset_bytes=dataset_bytes, 
                project_name=safe_project_name
            )
            
        print(f"   ✅ SUCCESS! Model: {model_url}")
        print("="*60 + "\n")
        
        return {
            "training_status": "success", 
            "model_url": model_url,
            "messages": [f"✅ Training complete! Model available at: {model_url}"]
        }
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        print("   💡 If this was a 'Deadline exceeded' error, your Mac likely went to sleep.")
        return {"training_status": "failed", "error": str(e)}