import streamlit as st
import os
from huggingface_hub import HfApi
import modal

# Ensure the Hugging Face token is set in your local environment (.env)
HF_TOKEN = os.environ.get("HF_TOKEN")

st.set_page_config(page_title="AI Showroom", page_icon="🤖", layout="wide")
st.title("🤖 AI Model Showroom")
st.write("Select a factory-trained brain and chat directly, or export it to run locally.")

# ==========================================
# FEATURE 1: DYNAMIC MODEL DISCOVERY
# ==========================================
@st.cache_data(ttl=60) # Caches the list for 60 seconds so it doesn't spam the API
def get_trained_models(username="Pineco04"):
    try:
        api = HfApi(token=HF_TOKEN)
        # Fetch all models authored by you
        models = api.list_models(author=username)
        # Extract just the repository names
        return [m.id for m in models]
    except Exception as e:
        st.error(f"Failed to connect to Hugging Face: {e}")
        return []

available_models = get_trained_models()

if not available_models:
    st.warning("No models found on Hugging Face. Run the Factory first!")
    st.stop()

# Dropdown to select the bot
selected_model = st.selectbox("🧠 Select a Model to Interact With:", available_models)

st.divider()

# ==========================================
# FEATURE 2 & 3: INFERENCE & OLLAMA EXPORT
# ==========================================
col1, col2 = st.columns([3, 1])

with col2:
    # Feature 3: The Ollama Export Button
    st.write("### Local Execution")
    if st.button("⬇️ Download for Ollama"):
        st.info(
            f"""**Ollama Export Instructions:**
            
To run `{selected_model}` locally utilizing your M4 MacBook's 16GB of unified memory, the model must be converted to a GGUF format. 

Currently, the factory outputs raw safetensors. Once GGUF export is added to the Training Agent, you will run:
```bash
ollama run {selected_model.split('/')[-1]}
"""
)
with col1:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Say something..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Connect to Modal for Inference
        with st.spinner("☁️ Pinging Modal GPU..."):
            try:
                # We dynamically look up the function deployed from your Factory!
                inference_func = modal.Function.from_name("inference-agent", "run_inference_on_cloud")
            
                # Run the inference remotely
                response = inference_func.remote(prompt, selected_model)
            
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            except Exception as e:
                st.error(f"Cloud Inference Failed: {e}")
