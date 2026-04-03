import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def data_node(state):
    print("\n--- 🏭 DATA AGENT STARTED ---")
    
    # 1. Load Dynamic Config
    topic = state.get("data_topic")
    style = state.get("data_style")
    target_count = state.get("dataset_size", 100)
    ref_content = state.get("dataset_content", "")[:2000] # Use scraped context!
    
    # 2. Dynamic Path
    folder = "training_data"
    os.makedirs(folder, exist_ok=True)
    filename = f"{topic.replace(' ', '_').lower()}_data.jsonl"
    file_path = os.path.join(folder, filename)
    
    # 3. Check Progress
    current_count = 0
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            current_count = sum(1 for line in f if line.strip())
            
    needed = target_count - current_count
    
    if needed <= 0:
        print(f"   ✅ Data Complete ({current_count}/{target_count})")
        return {"training_file_path": file_path, "generated_count": current_count}

    print(f"   🎭 Persona: {topic} | Needed: {needed}")
    
    # 4. Generate
    llm = ChatOpenAI(model="gpt-4o", temperature=0.8)
    
    # This prompt adapts to ANY topic using the scraped context
    prompt = ChatPromptTemplate.from_template(
        """You are a Synthetic Data Generator.
        
        TOPIC: {topic}
        STYLE: {style}
        REFERENCE CONTEXT (Use style/vocab from here if relevant): 
        {ref_content}
        
        TASK: Generate {batch_size} high-quality conversation pairs.
        FORMAT: JSON list [{{ "instruction": "...", "output": "..." }}]
        
        RULES:
        1. Direct speech only.
        2. NO "System" or "User" labels in the text strings.
        3. Match the requested style exactly.
        """
    )
    
    chain = prompt | llm | JsonOutputParser()
    
    # Safe batch size to prevent timeouts
    batch_size = min(10, needed)
    
    try:
        result = chain.invoke({
            "batch_size": batch_size,
            "topic": topic,
            "style": style,
            "ref_content": ref_content
        })
        
        # Write to file
        final_batch = result[:batch_size]
        with open(file_path, "a") as f:
            for entry in final_batch:
                f.write(json.dumps(entry) + "\n")
                
        new_total = current_count + len(final_batch)
        print(f"   ✅ Saved {len(final_batch)} rows. Total: {new_total}")
        
        return {
            "generated_count": new_total,
            "training_file_path": file_path
        }
        
    except Exception as e:
        print(f"   ❌ Generation Error: {e}")
        return {"generated_count": current_count} # Prevents crash, just retries