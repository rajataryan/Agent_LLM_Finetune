import modal
from modal import Secret

# --- FIX: Create a new App instance ---
# We cannot use .lookup() here because we are defining a new function.
app = modal.App("inference-agent") 

# Define the Inference Image (Lightweight)
inference_image = (
    modal.Image.debian_slim()
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "torch",
        "transformers",
        "huggingface_hub",
        "accelerate",
        "peft"
    )
)

@app.function(
    image=inference_image,
    gpu="A10G",
    secrets=[Secret.from_name("huggingface")]
)
def run_inference_on_cloud(prompt: str, model_id: str):
    """
    Downloads the fine-tuned model and generates a response.
    """
    import torch
    from unsloth import FastLanguageModel
    
    print(f"⚡ LOADING MODEL: {model_id}...")
    
    # 1. Load Model + Adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    
    # 2. Format Prompt (Alpaca Style)
    alpaca_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
    
    # 3. Generate
    inputs = tokenizer([alpaca_prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        use_cache=True,
        temperature=0.3, 
    )
    
    # 4. Decode
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Clean up the output
    response_clean = response.split("### Response:")[-1].strip()
    return response_clean

# --- MCP COMPATIBLE TOOL DEFINITION ---
def query_finetuned_model_tool(prompt: str, model_url: str):
    """
    MCP Tool: specific tool to query the sofa bot.
    """
    # Extract Repo ID from URL
    if "huggingface.co/" in model_url:
        repo_id = model_url.split("huggingface.co/")[-1]
    else:
        repo_id = model_url
        
    print(f"   🌩️ Calling Cloud Inference for: {repo_id}")
    
    # Run remotely on Modal
    # This will spin up the 'inference-agent' app temporarily to run the function
    with app.run():
        answer = run_inference_on_cloud.remote(prompt, repo_id)
        
    return answer