import os
import modal

# 1. Define the Cloud Environment
image = (
    modal.Image.debian_slim()
    .apt_install("git") 
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "torch",
        "torchvision", 
        "transformers",
        "datasets",
        "huggingface_hub",
        "trl",
        "accelerate",
        "peft"
    )
    .env({"HUGGINGFACE_TOKEN": os.environ.get("HUGGINGFACE_TOKEN", "")})
)

# 2. Define the App
app = modal.App("fine-tune-agent")

# 3. The GPU Function
@app.function(
    image=image,
    gpu="A10G",        
    timeout=3600,      
    secrets=[modal.Secret.from_name("huggingface")] 
)
def train_model_on_modal(data_content: str, config: dict):
    import os
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"🚀 CLOUD START: Training '{config.get('project_name')}'...")
    
    # A. Write Data
    with open("train.jsonl", "w") as f:
        f.write(data_content)
        
    # B. Load Model
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.get("base_model", "unsloth/llama-3-8b-bnb-4bit"),
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    # C. Format Data
    dataset = Dataset.from_json("train.jsonl")
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs      = examples["output"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}<|end_of_text|>"""
            texts.append(text)
        return { "text" : texts, }
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # D. Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # E. Train
    print("⚡ STARTING TRAINING RUN...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, 
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )
    trainer.train()
    
    # F. Save & Push
    print("💾 SAVING MODEL TO HUGGING FACE...")
    new_model_name = f"{config['project_name']}-lora"
    username = "Pineco04"  
    repo_id = f"{username}/{new_model_name}"
    hf_token = os.environ["HUGGINGFACE_TOKEN"]
    
    # --- THE FIX: Pass 'repo_id' (full address) instead of 'new_model_name' ---
    model.push_to_hub_merged(repo_id, tokenizer, save_method = "lora", token = hf_token)
    model.push_to_hub(repo_id, token = hf_token)
    
    final_url = f"https://huggingface.co/{repo_id}"
    print(f"✅ DONE! Model live at: {final_url}")
    
    return final_url