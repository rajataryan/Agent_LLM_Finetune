import modal

# Import the app first
from tools.training_tools import app, train_love_bot

modal.enable_output()

# Use @app.local_entrypoint() decorator
@app.local_entrypoint()
def main():
    # Read your actual training data
    data_file = "training_data/Conversational_AI_data.jsonl"
    
    print(f"📂 Reading training data from: {data_file}")
    
    with open(data_file, "rb") as f:
        data = f.read()
    
    print(f"✅ Loaded {len(data)} bytes ({len(data)/1024:.2f} KB)")
    print("🚀 Starting Modal training job...")
    print("   This may take 3-5 minutes (cold start + training)")
    print("-" * 60)
    
    # Call the remote function
    result = train_love_bot.remote(
        dataset_bytes=data,
        project_name="conversational-ai-bot"
    )
    
    print("-" * 60)
    print(f"🎉 SUCCESS!")
    print(f"✅ Model URL: {result}")
    print(f"\nYou can now use this model at: {result}")
