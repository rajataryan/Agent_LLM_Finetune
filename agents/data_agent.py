import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from state import AgentState

# --- CONFIGURATION ---
SAFE_BATCH_SIZE = 15

def data_node(state: AgentState):
    print("--- 🏭 DATA AGENT STARTED (HYBRID: OPENAI) ---")
    
    # 1. Get State
    topic = state.get("data_topic", "General Knowledge")
    style = state.get("data_style", "Helpful")
    target_count = state.get("dataset_size", 100)
    
    # 2. Setup File Path
    folder = "training_data"
    os.makedirs(folder, exist_ok=True)
    filename = f"{topic.replace(' ', '_')}_data.jsonl"
    file_path = os.path.join(folder, filename)
    
    # Count existing
    current_count = 0
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            current_count = sum(1 for line in f)
    
    needed = target_count - current_count
    
    if needed <= 0:
        return {
            "generated_count": current_count,
            "training_file_path": file_path,
            "training_status": "ready"
        }
    
    print(f"   🎯 Target: {target_count} | Needed: {needed}")
    
    # 3. Setup Model
    llm = ChatOpenAI(model="gpt-4o", temperature=0.9) # High temp for creativity
    
    # --- THE FIX IS HERE: NEW PROMPT ---
    prompt = ChatPromptTemplate.from_template(
        """You are creating a Roleplay Chatbot Dataset.
        The goal is to train a model to act as a specific character.
        
        CHARACTER PERSONA: {topic}
        TONE/STYLE: {style} (e.g., Shy, Romantic, Deeply in Love)
        
        TASK: Create {batch_size} conversation pairs between a User and this Character.
        
        FORMAT:
        Return a strictly valid JSON list.
        - "instruction": The User speaking directly to the character (Questions, flirting, greetings).
        - "output": The Character responding directly to the User.
        
        BAD EXAMPLE (Do NOT do this):
        {{"instruction": "Write a love letter", "output": "Dear Love..."}}
        
        GOOD EXAMPLE (DO THIS):
        {{"instruction": "Do you love me?", "output": "I... I have loved you since the first moment I saw you."}}
        {{"instruction": "Hi!", "output": "Oh! Hi... I was just hoping you would message me."}}
        
        CRITICAL RULES:
        1. No descriptive instructions. Direct speech only.
        2. The character is SHY, WARM, and IN LOVE.
        3. Use "..." and stammers to show shyness if needed.
        """
    )
    
    chain = prompt | llm | JsonOutputParser()
    
    # 4. Generation Loop
    new_data = []
    
    while len(new_data) < needed:
        remaining = needed - len(new_data)
        current_batch_size = min(SAFE_BATCH_SIZE, remaining)
        
        print(f"   ⏳ Generating batch of {current_batch_size}...")
        
        try:
            # We don't even need the scraped content for this specific request
            # We rely on GPT-4o's creative writing
            batch_result = chain.invoke({
                "batch_size": current_batch_size,
                "topic": topic,
                "style": style
            })
            
            if isinstance(batch_result, list):
                new_data.extend(batch_result)
                print(f"      ✅ Success. Total gathered this run: {len(new_data)}")
                
                with open(file_path, "a") as f:
                    for entry in batch_result:
                        f.write(json.dumps(entry) + "\n")
            
        except Exception as e:
            print(f"      ❌ Batch failed: {str(e)[:100]}... (Skipping)")
            if len(new_data) == 0: break 

    return {
        "generated_count": current_count + len(new_data),
        "training_file_path": file_path,
        "training_status": "ready" if (current_count + len(new_data)) >= target_count else "pending"
    }