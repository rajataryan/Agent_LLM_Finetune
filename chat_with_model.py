import modal

# 1. Define the Cloud Environment
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "colorama" 
    )
)

app = modal.App("girlfriend-bot", image=image)

# 2. Define the Model Class
@app.cls(gpu="A10G", timeout=600)
class ModelInference:
    
    # ⚠️ CRITICAL FIX: We use @modal.enter() instead of __enter__
    # This ensures the model loads automatically when the container starts.
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_id = "Pineco04/romantic-girlfriend-bot"
        
        print(f"📥 Downloading/Loading model: {model_id}...")

        # Load the Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load the Full Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("✅ Model loaded successfully!")

    @modal.method()
    def generate_response(self, user_input):
        import torch
        
        # Create the prompt structure
        messages = [
            {"role": "system", "content": "You are a loving, romantic girlfriend. You answer softly and care about the user's day."},
            {"role": "user", "content": user_input},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        response = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return response

# 3. Local Entrypoint (Runs on your Mac)
@app.local_entrypoint()
def main():
    import colorama
    from colorama import Fore, Style
    colorama.init()

    print(Fore.CYAN + "Initializing connection to your Girlfriend Bot..." + Style.RESET_ALL)
    
    bot = ModelInference()

    # Pre-warm the container (Optional, triggers the download immediately)
    # print("Wake up call...") 
    # bot.generate_response.remote("Hello") 

    while True:
        try:
            # ⚠️ UX FIX: Putting the prompt INSIDE input() so it doesn't get overwritten
            user_input = input(f"\n{Fore.GREEN}You: {Style.RESET_ALL}")
            
            if user_input.lower() in ["exit", "quit"]:
                print(Fore.RED + "Ending chat. Goodbye!" + Style.RESET_ALL)
                break

            print(Fore.MAGENTA + "💕 She is typing..." + Style.RESET_ALL)
            
            # This calls the code running in the cloud
            response = bot.generate_response.remote(user_input)
            
            print(f"{Fore.MAGENTA}Girlfriend:{Style.RESET_ALL} {response}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n{Fore.RED}Error occurred: {e}{Style.RESET_ALL}")
            break