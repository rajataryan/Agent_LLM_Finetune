import streamlit as st
import os
import sys
from dotenv import load_dotenv
from main import build_graph

# Force load environment variables
load_dotenv(override=True)

# Verify Modal is available
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    st.error("❌ Modal SDK not installed. Run: `pip install modal`")

# Page config
st.set_page_config(page_title="AI Fine-Tuning Factory", layout="wide")

st.title("🏭 Agentic AI Factory")
st.markdown("Enter a goal below. This pipeline will scrape data, generate a dataset, and fine-tune a Llama 3 model for you.")

# Verify prerequisites
with st.sidebar:
    st.header("⚙️ System Status")
    
    # Check HuggingFace Token
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        st.success(f"✅ HF Token: {hf_token[:10]}...")
    else:
        st.error("❌ HUGGINGFACE_TOKEN missing")
        st.caption("Add to `.env` file")
    
    # Check Modal
    if MODAL_AVAILABLE:
        st.success("✅ Modal SDK installed")
        try:
            # Check if Modal is authenticated
            result = os.system("modal token verify > /dev/null 2>&1")
            if result == 0:
                st.success("✅ Modal authenticated")
            else:
                st.warning("⚠️ Modal not authenticated")
                st.caption("Run: `modal token new`")
        except:
            st.warning("⚠️ Could not verify Modal auth")
    else:
        st.error("❌ Modal not installed")
    
    st.divider()
    st.caption("Made with ❤️ using LangGraph + Modal")

# Input section
col1, col2 = st.columns([2, 1])

with col1:
    user_goal = st.text_area(
        "What kind of bot do you want?", 
        "I want a polite customer support bot for 'ComfyCloud Sofas' that handles questions about returns, shipping, and fabric care.",
        height=150
    )

with col2:
    st.info("⚙️ **Settings**")
    project_name = st.text_input(
        "Project Name (HuggingFace)", 
        "comfycloud-sofa-bot",
        help="Must be lowercase, use hyphens instead of spaces"
    )
    # --- UPDATED INPUT FIELD ---
    dataset_size = st.number_input(
        "Training Examples",
        min_value=10,       # Start at 10
        max_value=1200,     # Allow up to 1200
        value=50,           # Default reasonable starting value
        step=10,            # Increment by 10
        help="More examples = better model (but longer training)"
    )
    # ---------------------------

# Run button
if st.button("🚀 Start Production Line", type="primary", disabled=not MODAL_AVAILABLE):
    
    if not hf_token:
        st.error("❌ Cannot start: HUGGINGFACE_TOKEN not found in environment")
        st.stop()
    
    # Initialize graph
    graph = build_graph()
    
    initial_state = {
        "messages": [],
        "user_goal": user_goal,
        "project_name": project_name,
        "dataset_size": dataset_size,
        "generated_count": 0,
        "training_status": "pending"
    }
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Detailed status container
    with st.container():
        st.write("--- 🏁 **Pipeline Started** ---")
        current_task = st.empty()
        detail_expander = st.expander("📋 Detailed Logs", expanded=False)
        
        try:
            # --- UPDATED RECURSION LIMIT ---
            # Increased to 1000 to handle large dataset generation loops
            events = graph.stream(initial_state, config={"recursion_limit": 1000})
            
            step = 0
            total_steps = 5  # intake, browser, data, training, finalizer
            
            for event in events:
                for node, state in event.items():
                    step += 1
                    progress = min(step / total_steps, 1.0)
                    progress_bar.progress(progress)
                    
                    if node == "intake_agent":
                        status_text.text("🧠 Analyzing request...")
                        current_task.info("🧠 **Intake Agent:** Understanding your requirements...")
                        with detail_expander:
                            st.write(f"**Topic:** {state.get('data_topic', 'N/A')}")
                            st.write(f"**Style:** {state.get('data_style', 'N/A')}")
                            
                    elif node == "browser_agent":
                        status_text.text("🌐 Scraping web data...")
                        current_task.info("🌐 **Browser Agent:** Gathering training data from the web...")
                        urls = state.get('site_list', [])
                        if urls:
                            with detail_expander:
                                st.success(f"✅ Scraped {len(urls)} pages")
                                for url in urls[:3]:  # Show first 3
                                    st.caption(f"- {url}")
                        
                    elif node == "data_agent":
                        count = state.get('generated_count', 0)
                        target = state.get('dataset_size', 10)
                        status_text.text(f"📝 Generating training data ({count}/{target})...")
                        current_task.info(f"📝 **Data Agent:** Creating training examples ({count}/{target})...")
                        if count >= target:
                            with detail_expander:
                                st.success(f"✅ Generated {count} training samples")
                                file_path = state.get('training_file_path')
                                if file_path:
                                    st.caption(f"📁 Saved to: {file_path}")
                            
                    elif node == "training_agent":
                        status = state.get('training_status')
                        status_text.text("⚡ Training model on GPU...")
                        
                        if status == "success":
                            url = state.get('model_url')
                            current_task.success("✅ **Training Complete!**")
                            st.balloons()
                            
                            st.markdown(f"""
                            ### 🎉 Your Model is Ready!
                            
                            **Model URL:** [{url}]({url})
                            
                            You can now use this model for inference!
                            """)
                            
                            st.session_state.final_model_url = url
                            
                        elif status == "failed":
                            error = state.get('error', 'Unknown error')
                            current_task.error(f"❌ Training failed: {error}")
                            with detail_expander:
                                st.error(f"**Error Details:** {error}")
                        else:
                            current_task.warning("⚡ **Training Agent:** Starting GPU training (2-5 minutes)...")
                            with detail_expander:
                                st.info("This step involves:\n- Cold start (~1-2 min)\n- Model loading (~30s)\n- Training (~2-3 min)")
                            
                    elif node == "finalizer_agent":
                        status_text.text("🤖 Testing model...")
                        response = state.get('final_response')
                        current_task.success("🤖 **Finalizer Agent:** Model test complete!")
                        
                        st.divider()
                        st.markdown("### 💬 Model Test Response")
                        st.info(response)
                        
            progress_bar.progress(1.0)
            status_text.text("✅ Pipeline complete!")

        except Exception as e:
            st.error(f"❌ Pipeline failed: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(str(e))
            
            # Common troubleshooting
            st.markdown("""
            **Common Issues:**
            - Modal not authenticated: Run `modal token new`
            - Missing HF token: Add to `.env` file
            - Network issues: Check internet connection
            - Ollama not running: Start with `ollama serve`
            """)

# Show previous results if available
if 'final_model_url' in st.session_state:
    st.divider()
    st.markdown("### 📌 Last Trained Model")
    st.success(f"[{st.session_state.final_model_url}]({st.session_state.final_model_url})")