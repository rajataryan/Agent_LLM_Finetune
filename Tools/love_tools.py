import os
import modal
from dotenv import load_dotenv

load_dotenv()

# 1. NEW APP NAME (Fresh Start)
app = modal.App("love-bot-factory-v2")

# 2. NEW IMAGE (Clean Build)
# We add a timestamp to the env var to ensure it's always unique
image = (
    modal.Image.from_registry("unsloth/meta-llama-3.1-8b-bnb-4bit:latest")
    .env({"CACHE_BUST": "FRESH_FILE_V1"}) 
    .pip_install("huggingface_hub")
)

# 3. FUNCTION
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    secrets=[modal.Secret.from_dict({"HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN")})]
)
def train_love_bot(dataset_bytes: bytes, project_name: str):
    import os
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset
    from huggingface_hub import HfApi

    print(f"🚀 STARTING CLOUD TRAINING: {project_name}")

    # Write data
    data_path = "dataset.jsonl"
    with open(data_path, "wb") as f:
        f.write(dataset_bytes)

    # Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

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

    # Format Data
    dataset = load_dataset("json", data_files=data_path, split="train")

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]
        texts = []
        for i, o in zip(instructions, outputs):
            text = f"<|start_header_id|>user<|end_header_id|>\n\n{i}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{o}<|eot_id|>"
            texts.append(text)
        return {"text": texts}

    # Train
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = False,
        formatting_func = formatting_prompts_func,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, 
            learning_rate = 2e-4,
            fp16 = True,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",
        ),
    )

    print("   ⚡ Training started...")
    trainer.train()

    # Save
    print("   💾 Saving to Hugging Face...")
    token = os.environ["HUGGINGFACE_TOKEN"]
    api = HfApi()
    user = api.whoami(token=token)["name"]
    repo_id = f"{user}/{project_name}"
    
    model.push_to_hub_merged(repo_id, tokenizer, save_method="lora", token=token)
    
    return f"https://huggingface.co/{repo_id}"