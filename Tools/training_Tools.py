import os
import modal

# Enable output for debugging
modal.enable_output()

# 1. FIXED: Use a static app name. 
# Do not use time.time() here, it causes issues with connection state.
app = modal.App("finetune-factory")

# IMAGE DEFINITION
image = (
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
    image=image,
    gpu="A10G",
    timeout=60 * 60, # 1 hour max server timeout
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_generic_model(dataset_bytes: bytes, project_name: str):
    import os
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset
    from huggingface_hub import HfApi

    print(f"🚀 CLOUD TRAINING STARTED: {project_name}")
    
    # Write dataset
    data_path = "/tmp/dataset.jsonl"
    with open(data_path, "wb") as f:
        f.write(dataset_bytes)
    print(f"✅ Dataset written: {len(dataset_bytes)} bytes")

    # Load model
    print("📦 Loading Llama 3.1 8B (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Apply LoRA
    print("🔧 Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Load dataset
    print("📊 Loading training data...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"✅ Loaded {len(dataset)} examples")

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]
        texts = []
        for i, o in zip(instructions, outputs):
            text = f"<|start_header_id|>user<|end_header_id|>\n\n{i}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{o}<|eot_id|>"
            texts.append(text)
        return texts 

    # Train
    print("🏋️ Training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        formatting_func=formatting_prompts_func,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=False, 
            bf16=True, 
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="/tmp/outputs",
            report_to="none",
        ),
    )
    
    trainer.train()
    print("✅ Training complete!")

    # Push to Hub
    print("📤 Uploading to HuggingFace...")
    token = os.environ["HUGGINGFACE_TOKEN"]
    api = HfApi()
    user = api.whoami(token=token)["name"]
    repo_id = f"{user}/{project_name}"
    
    model.push_to_hub_merged(repo_id, tokenizer, save_method="lora", token=token)
    
    url = f"https://huggingface.co/{repo_id}"
    print(f"🎉 SUCCESS! {url}")
    return url