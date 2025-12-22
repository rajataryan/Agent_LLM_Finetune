import modal
import sys

# Force output before importing
modal.enable_output()

print("=" * 60)
print("🚀 MODAL TRAINING TEST")
print("=" * 60)

# Import after enabling output
from tools.training_tools import train_love_bot, app

# Read training data
print("\n📂 Reading training data...")
try:
    with open("training_data/Conversational_AI_data.jsonl", "rb") as f:
        data = f.read()
    print(f"✅ Loaded {len(data)} bytes ({len(data)/1024:.2f} KB)")
except FileNotFoundError:
    print("❌ Error: training_data/Conversational_AI_data.jsonl not found!")
    sys.exit(1)

print("\n🚀 Dispatching to Modal...")
print("⏳ This may take 5-10 minutes on first run (building image)")
print("=" * 60)

try:
    # Run with context manager
    with app.run():
        print("\n✅ App running, calling train_love_bot.remote()...")
        result = train_love_bot.remote(
            dataset_bytes=data,
            project_name="conversational-ai-test"
        )
    
    print("\n" + "=" * 60)
    print("🎉 SUCCESS!")
    print(f"✅ Model URL: {result}")
    print("=" * 60)

except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user")
    sys.exit(1)
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ TRAINING FAILED")
    print(f"Error: {e}")
    print("=" * 60)
    
    # Print more details
    import traceback
    traceback.print_exc()
    sys.exit(1)