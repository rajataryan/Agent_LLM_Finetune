import gradio as gr
import os
from main import build_graph  # Import the graph builder we created in main.py

def run_pipeline(user_goal):
    """
    Runs the agentic pipeline and streams logs to the UI.
    """
    # 1. Validation
    if not user_goal.strip():
        yield "⚠️ Please enter a goal first!", None, None
        return

    # 2. Setup
    logs = "🚀 STARTING PIPELINE...\n"
    yield logs, None, None  # Update UI immediately

    try:
        # Build graph using the logic from main.py
        graph = build_graph()
        
        # Initial State
        initial_state = {
            "messages": [],
            "user_goal": user_goal,
            # We let the Intake Agent determine the rest (Project Name, Size, etc.)
        }
        
        # 3. Execution Loop
        # We use a generator (yield) to stream updates to the UI in real-time
        events = graph.stream(initial_state, config={"recursion_limit": 50})
        
        final_file_path = None
        
        for event in events:
            for node_name, values in event.items():
                # --- Format the Log Message ---
                logs += f"\n✅ FINISHED NODE: {node_name}\n"
                
                # Extract details based on which agent finished
                if "project_name" in values and node_name == "intake_agent":
                    logs += f"   👉 Project: {values['project_name']}\n"
                    logs += f"   👉 Topic: {values['data_topic']} (Size: {values['dataset_size']})\n"
                
                if "site_list" in values and node_name == "browser_agent":
                    count = len(values['site_list'])
                    logs += f"   👉 Found {count} URLs to process.\n"
                    
                if "training_file_path" in values:
                    final_file_path = values['training_file_path']
                    logs += f"   🎉 SUCCESS: File saved.\n"

                # Update the log window
                yield logs, None, None
        
        # 4. Final Result Display
        if final_file_path and os.path.exists(final_file_path):
            logs += "\n🏁 PIPELINE FINISHED SUCCESSFULLY."
            
            # Read the first few lines of the file for preview
            with open(final_file_path, "r") as f:
                preview_content = "".join([next(f) for _ in range(5)]) # First 5 lines
            
            yield logs, final_file_path, preview_content
        else:
            logs += "\n❌ Error: No output file was generated."
            yield logs, None, None

    except Exception as e:
        logs += f"\n❌ CRITICAL ERROR: {str(e)}"
        yield logs, None, None

# --- UI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Fine-Tuning Dataset Generator")
    gr.Markdown("Describe your dream LLM, and this agentic team will scrape the web and build a training dataset for you.")
    
    with gr.Row():
        with gr.Column(scale=1):
            goal_input = gr.Textbox(
                label="What kind of chatbot do you want?", 
                placeholder="e.g., A customer support bot for Apple iPads that is very polite",
                lines=3
            )
            run_btn = gr.Button("🚀 Launch Agents", variant="primary")
        
        with gr.Column(scale=2):
            log_output = gr.Textbox(
                label="Agent Activity Logs", 
                lines=15, 
                interactive=False,
                autoscroll=True
            )
    
    with gr.Row():
        file_output = gr.File(label="Download Dataset (.jsonl)")
        preview_output = gr.Code(label="Data Preview (First 5 Lines)", language="json")

    # Connect the button to the function
    run_btn.click(
        fn=run_pipeline, 
        inputs=goal_input, 
        outputs=[log_output, file_output, preview_output]
    )

if __name__ == "__main__":
    demo.launch()