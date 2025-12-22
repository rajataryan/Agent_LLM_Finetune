import os
import modal
import uuid
from dotenv import load_dotenv

load_dotenv(override=True)

# 1. GENERATE UNIQUE ID
# This guarantees a fresh build every time.
run_id = str(uuid.uuid4())[:8]
app_name = f"love-bot-fix-{run_id}"

print(f"🚀 STARTING JOB: {app_name}")
print(f"   (Unique Build ID: {run_id})")

if not os.getenv("HUGGINGFACE_TOKEN"):
    print("❌ Error: HUGGINGFACE_TOKEN not found in .env")
    exit(1)

app = modal.App(app_name)

# 2. IMAGE DEFINITION
image = (
    # Use the Official Unsloth Image (Fastest Setup)
    modal.Image.from_registry("unsloth/meta-llama-3.1-8b-bnb-4bit:latest")
    # ------------------------------------------------------------------
    # 🛡️ CACHE BUSTER: This specific line forces a new build hash
    # ------------------------------------------------------------------
    .run_commands(f"echo 'Build ID: {run_id}'") 
    .pip_install(
        "huggingface_hub",
        "trl",
        "datasets"
    )
)

# 3. TRAINING FUNCTION
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    secrets=[modal.Secret.from_dict({"HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN")})]
)
def train_script(dataset_bytes: bytes, project_name: str):
    import os
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset
    from huggingface_hub import HfApi

    print(f"🚀 CLOUD STARTED: Training {project_name}...")

    # Write Data
    with open("dataset.jsonl", "wb") as f:
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
    dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

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

    print("   ⚡ Training...")
    trainer.train()

    # Save
    token = os.environ["HUGGINGFACE_TOKEN"]
    api = HfApi()
    user = api.whoami(token=token)["name"]
    repo_id = f"{user}/{project_name}"
    
    print(f"   💾 Pushing to: {repo_id}")
    model.push_to_hub_merged(repo_id, tokenizer, save_method="lora", token=token)
    
    return f"https://huggingface.co/{repo_id}"

# 4. RUN LOCAL
if __name__ == "__main__":
    modal.enable_output()
    
    import glob
    files = glob.glob("training_data/*.jsonl")
    
    if not files:
        print("❌ Error: No .jsonl file found!")
        exit(1)
        
    file_path = files[0]
    print(f"📂 Found data file: {file_path}")
    
    with open(file_path, "rb") as f:
        data = f.read()
    
    print(f"   Sending {len(data)} bytes to the cloud...")
    
    with app.run():
        url = train_script.remote(data, "love-bot-final")
        print(f"\n✅ SUCCESS! Your Love Bot is here: {url}")