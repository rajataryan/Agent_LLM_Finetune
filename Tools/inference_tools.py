import modal
from modal import Secret

# Enable output so we can see the image build process!
modal.enable_output()

# --- Create a new App instance ---
app = modal.App("inference-agent") 

# --- CRITICAL FIX: Force CUDA 12.1 GPU Installations & Remove Rogue Dependencies ---
inference_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget")
    # 1. Force pip to grab the GPU versions, NOT the CPU versions
    .pip_install(
        "torch", 
        "torchvision", 
        "torchaudio", 
        index_url="https://download.pytorch.org/whl/cu121"
    )
    # 2. Install the Hugging Face stack
    .pip_install(
        "transformers",
        "datasets",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "huggingface-hub"
    )
    # 3. Install Unsloth
    .pip_install("unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
    # 4. Strip out the incompatible torchao library so it stops crashing!
    .run_commands("pip uninstall -y torchao")
)

@app.function(
    image=inference_image,
    gpu="A10G",
    # Ensure this matches the secret name used in training_tools.py
    secrets=[Secret.from_name("huggingface-secret")] 
)
def run_inference_on_cloud(prompt: str, model_id: str):
    """
    Logs in globally, then lets Unsloth download the base model and adapter automatically.
    """
    import os
    from huggingface_hub import login
    import torch
    from unsloth import FastLanguageModel
    
    # --- 1. GLOBAL VIP PASS ---
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("🔑 Logging into Hugging Face globally...")
        login(token=hf_token)
    else:
        print("⚠️ WARNING: HF_TOKEN not found!")
    
    print(f"⚡ LOADING FULL MODEL: {model_id}...")
    
    # --- 2. ONE-STEP MAGIC ---
    # Because we are logged in, Unsloth can see the private repo, 
    # find the base model automatically, and merge your adapter!
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id, 
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    
    # 3. Format Prompt (Llama 3 Style)
    llama_style_prompt = f"""<|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # 4. Generate
    inputs = tokenizer([llama_style_prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        use_cache=True,
        temperature=0.3, 
    )
    
    # 5. Decode
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