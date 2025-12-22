import gradio as gr
import os
from dotenv import load_dotenv
from tools.inference_tools import query_finetuned_model_tool

# Load environment variables (API keys)
load_dotenv()

def generate_response(prompt, model_url):
    """
    This function creates the bridge between the UI and the Cloud GPU.
    """
    if not prompt:
        return "⚠️ Please enter a question."
    if not model_url:
        return "⚠️ Please provide a Model URL."
    
    # Extract the Repo ID if the user pastes a full URL
    if "https://" in model_url:
        model_id = model_url.split("huggingface.co/")[-1]
    else:
        model_id = model_url
        
    try:
        # Call the inference tool we built earlier
        return query_finetuned_model_tool(prompt, model_id)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# --- BUILD THE USER INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛋️ ComfyCloud Sofa Bot (Chat Interface)")
    gr.Markdown("Chat with your fine-tuned model running on a Modal Cloud GPU.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 1. Configuration Input
            model_input = gr.Textbox(
                label="Hugging Face Model ID", 
                value="Pineco04/comfycloud-sofa-bot-lora", 
                info="Paste your model repo here (e.g. username/project-name-lora)"
            )
            
        with gr.Column(scale=2):
            # 2. Chat Inputs
            user_input = gr.Textbox(
                label="Your Question", 
                placeholder="Ask about sofas, shipping, or returns...",
                lines=2
            )
            submit_btn = gr.Button("🚀 Generate Answer", variant="primary")
            
            # 3. Output Display
            output_box = gr.Textbox(
                label="Model Response", 
                interactive=False, 
                lines=10
            )

    # 4. Examples (Defined AFTER inputs are created)
    gr.Examples(
        examples=[
            ["Do you offer white glove delivery?"],
            ["What is your return policy?"],
            ["How do I clean the velvet fabric?"],
            ["Will the sofa fit through a 30 inch door?"]
        ],
        inputs=user_input
    )

    # 5. Connect Buttons to Logic
    submit_btn.click(
        fn=generate_response, 
        inputs=[user_input, model_input], 
        outputs=output_box
    )
    
    user_input.submit(
        fn=generate_response, 
        inputs=[user_input, model_input], 
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch()