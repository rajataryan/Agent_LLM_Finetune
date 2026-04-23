import modal
from modal import Secret

# Enable output so we can see the image build process!
modal.enable_output()

# --- Create a new App instance ---
app = modal.App("inference-agent") 

# --- CRITICAL FIX: Match the Training Image Exactly ---
inference_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget")
    .pip_install(
        "transformers",
        "datasets",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "huggingface-hub",
    )
    .pip_install("unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
    .pip_install("torchvision")
)

@app.function(
    image=inference_image,
    gpu="A10G",
    # Ensure this matches the secret name used in training_tools.py
    secrets=[Secret.from_name("huggingface-secret")] 
)
def run_inference_on_cloud(prompt: str, model_id: str):
    """
    Downloads the base model, attaches the fine-tuned LoRA adapter, and generates a response.
    """
    import os
    from huggingface_hub import login
    import torch
    from unsloth import FastLanguageModel
    
    # --- NEW: Log into Hugging Face to access private models! ---
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("🔑 Logging into Hugging Face...")
        login(token=hf_token)
    else:
        print("⚠️ WARNING: HF_TOKEN not found in Modal environment!")
    
    print(f"⚡ LOADING BASE MODEL...")
    
    # 1. Load Base Model First (Llama 3.1 8B)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit", 
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # 1.5 Attach your custom LoRA personality
    print(f"🔗 Attaching custom LoRA adapter: {model_id}")
    model.load_adapter(model_id)
    
    FastLanguageModel.for_inference(model)
    
    # 2. Format Prompt (Llama 3 Style)
    llama_style_prompt = f"""<|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # 3. Generate
    inputs = tokenizer([llama_style_prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        use_cache=True,
        temperature=0.3, 
    )
    
    # 4. Decode
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Clean up the output to return ONLY the assistant's part
    if "assistant" in response:
        response_clean = response.split("assistant")[-1].strip()
    else:
        response_clean = response
        
    return response_clean

# --- MCP COMPATIBLE TOOL DEFINITION ---
def query_finetuned_model_tool(prompt: str, model_url: str):
    """
    MCP Tool: specific tool to query the fine-tuned bot.
    """
    # Extract Repo ID from URL
    if "huggingface.co/" in model_url:
        repo_id = model_url.split("huggingface.co/")[-1]
    else:
        repo_id = model_url
        
    print(f"   🌩️ Calling Cloud Inference for: {repo_id}")
    
    # Run remotely on Modal
    with app.run():
        answer = run_inference_on_cloud.remote(prompt, repo_id)
        
    return answer