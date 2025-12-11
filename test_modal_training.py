import os
import json
from agents.training_agent import training_node
from state import AgentState

def create_sofa_data_file(filename="sofa_data.jsonl"):
    """Creates a dataset for a fictional Sofa Company Support Bot."""
    
    # 10 Custom Examples: Support policies for 'ComfyCloud Sofas'
    data = [
        {"instruction": "Do you offer white glove delivery?", "output": "Yes! For a flat fee of $150, our White Glove team will deliver, unpack, assemble your sofa in your room of choice, and remove all packaging materials."},
        {"instruction": "My doorway is only 30 inches wide. Will the Cloud Sectional fit?", "output": "The Cloud Sectional ships in modular boxes that are approximately 28 inches wide. It is designed specifically to fit through standard narrow doorways and tight stairwells."},
        {"instruction": "How do I clean the velvet fabric if I spill coffee?", "output": "For velvet, blot (do not rub) the spill immediately with a clean, dry cloth. For stains, use a water-free solvent or dry cleaning product. Avoid using water directly on velvet as it may leave a ring."},
        {"instruction": "What is your return policy?", "output": "We offer a 365-day trial period. If you are not satisfied, you can return the sofa for a full refund, minus return shipping costs. Returns must be in original packaging."},
        {"instruction": "Are your fabrics pet-friendly?", "output": "Yes, our 'Performance Weave' and 'Microfiber' options are highly resistant to scratches and stains, making them perfect for homes with cats or dogs."},
        {"instruction": "Does the warranty cover sagging cushions?", "output": "Our 5-year warranty covers structural defects in the frame and suspension. Cushions naturally soften over time, which is considered normal wear and tear, but we offer replacement cushion cores at cost."},
        {"instruction": "Can I swap the chaise from left to right?", "output": "Absolutely. All ComfyCloud sectionals are fully modular. You can move the chaise unit to either side simply by unlocking the connector clips underneath."},
        {"instruction": "How long does shipping take to New York?", "output": "In-stock fabrics ship within 3-5 business days. Custom orders take 8-12 weeks. Transit time to New York is typically 2-4 days once it leaves our warehouse."},
        {"instruction": "Do you offer financing options?", "output": "Yes, we partner with Affirm to offer 0% APR financing for up to 24 months for qualified buyers. You can select this option at checkout."},
        {"instruction": "Where are your sofas manufactured?", "output": "Our frames are hand-crafted in North Carolina, USA, using sustainably sourced kiln-dried hardwood. Our fabrics are sourced from premium mills in Italy and Belgium."}
    ]
    
    # Repeat data to give the trainer enough volume to run for a few steps
    full_data = data * 10 
    
    folder = "training_data"
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    
    with open(file_path, "w") as f:
        for entry in full_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Created Sofa Support dataset at: {file_path}")
    return file_path

if __name__ == "__main__":
    print("🧪 STARTING MODAL UNIT TEST (SOFA EDITION)...")
    
    # 1. Setup Data
    file_path = create_sofa_data_file()
    
    # 2. Mock State
    mock_state = {
        "training_file_path": file_path,
        "base_model": "unsloth/llama-3-8b-bnb-4bit",
        "project_name": "comfycloud-sofa-bot", # <--- Custom Name
        
        # Required TypedDict fields (unused here)
        "messages": [],
        "user_goal": "Train Sofa Support Bot",
        "data_topic": "Customer Support",
        "data_style": "Polite",
        "dataset_size": 100,
        "language": "English",
        "site_list": [],
        "dataset_content": "",
        "generated_count": 100,
        "job_id": "",
        "training_status": "ready",
        "model_url": "",
        "status": "testing",
        "error": None
    }
    
    # 3. Trigger Training
    print("⚡ Calling training_node()...")
    result = training_node(mock_state)
    
    # 4. Report
    print("\n--- 🧪 TEST RESULTS ---")
    if result.get("training_status") == "success":
        print("✅ SUCCESS!")
        print(f"🔗 Model URL: {result.get('model_url')}")
    else:
        print("❌ FAILED.")
        print(f"Error: {result.get('error')}")