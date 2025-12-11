import json
import math
# --- BACK TO OPENAI ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, SystemMessage
from state import AgentState
from tools.file_tools import save_to_jsonl
from dotenv import load_dotenv

load_dotenv(override=True)

# OpenAI doesn't need "system prompt engineering" as much as local models, 
# but strict rules help quality.
DATA_SYSTEM_PROMPT = """
You are a Dataset Generator for fine-tuning LLMs.
Your goal is to create realistic "Instruction -> Response" pairs based on the provided text.

OUTPUT RULES:
1. Output MUST be valid JSONL.
2. Each line must be a self-contained JSON object.
3. REQUIRED KEYS: "instruction" and "output".
4. NO MARKDOWN. NO INTROS.

EXAMPLE:
{"instruction": "...", "output": "..."}
{"instruction": "...", "output": "..."}
"""

def data_node(state: AgentState):
    print("--- 🏭 DATA AGENT STARTED (HYBRID: OPENAI) ---")
    
    # 1. Get Context
    raw_text = state.get("dataset_content", "")
    topic = state.get("data_topic", "General")
    style = state.get("data_style", "Normal")
    target_total = state.get("dataset_size", 50) 
    
    if not raw_text:
        return {"error": "No raw text found!", "status": "failed"}

    # 2. Configure OpenAI (The "F1 Car")
    # We use response_format={"type": "json_object"} to force valid JSON
    llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0.5,
        model_kwargs={"response_format": {"type": "json_object"}} 
    )

    BATCH_SIZE = 50 
    truncated_text = raw_text[:50000] # OpenAI context window is huge, we can use more text
    all_generated_lines = []
    
    num_batches = math.ceil(target_total / BATCH_SIZE)
    print(f"   🎯 Target: {target_total} examples")
    print(f"   🔄 Strategy: {num_batches} batches via GPT-4o")

    for i in range(num_batches):
        print(f"   ⏳ Batch {i+1}/{num_batches}...")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=DATA_SYSTEM_PROMPT),
            ("human", """
            SOURCE TEXT:
            {text}
            
            TASK:
            Generate {batch_size} training examples related to '{topic}'.
            Style: {style}
            
            Output a JSON object with a key "examples" containing the list of objects.
            """)
        ])
        
        msg = prompt.format_messages(
            topic=topic, 
            style=style, 
            text=truncated_text, 
            batch_size=BATCH_SIZE
        )
        
        try:
            response = llm.invoke(msg).content
            
            # Parse the JSON object returned by OpenAI
            data = json.loads(response)
            
            # Extract the list (OpenAI usually wraps it if requested)
            # We handle both wrapped "examples": [] and raw list cases
            if "examples" in data:
                items = data["examples"]
            else:
                items = data
            
            # Convert back to JSONL lines
            valid_lines = [json.dumps(item) for item in items]
            
            if valid_lines:
                all_generated_lines.extend(valid_lines)
                print(f"      ✅ Batch {i+1} complete. Got {len(valid_lines)} pairs.")
            
        except Exception as e:
            print(f"      ❌ Batch {i+1} failed: {e}")
            continue

    if not all_generated_lines:
        return {"error": "Failed to generate data", "status": "failed"}

    # Trim to exact size
    final_dataset = all_generated_lines[:target_total]
    
    final_content = "\n".join(final_dataset)
    filename = f"{topic.replace(' ', '_')}_data.jsonl"
    file_path = save_to_jsonl(final_content, filename)
    
    return {
        "training_file_path": file_path,
        "status": "ready_to_train",
        "messages": [AIMessage(content=f"Data generation complete. Saved {len(final_dataset)} examples to {file_path}")]
    }