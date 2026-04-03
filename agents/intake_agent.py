from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from state import AgentState
from dotenv import load_dotenv
# Import the configuration schema from your tools
from tools.intake_tools import ProjectConfiguration 

load_dotenv(override=True)

# We use Llama 3.1 (8B) here because it follows strict JSON schemas better than the small 3.2 model
llm = ChatOllama(model="llama3.1:8b", temperature=0)

INTAKE_SYSTEM_PROMPT = """
You are an AI Architect Intake Agent.
Your job is to analyze a user's request and structure it into a project configuration.
If the user writes in a specific language (e.g., Hindi, Spanish), detect it and set the 'language' field.
"""

def intake_node(state: AgentState):
    print("--- 🧠 INTAKE AGENT STARTED (LOCAL) ---")
    
    # 1. Capture the value already set by the GUI/Main script
    gui_dataset_size = state.get("dataset_size")
    
    # 2. Get User Input
    messages = state.get("messages", [])
    user_input = messages[-1].content if messages else state.get("user_goal", "General LLM Fine-Tuning")

    # 3. Configure LLM for Structured Output
    structured_llm = llm.with_structured_output(ProjectConfiguration)
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTAKE_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    # 4. Execute LLM to get Topic, Style, etc.
    chain = prompt | structured_llm
    config = chain.invoke({"input": user_input})
    
    # 5. PRIORITY LOGIC
    # If gui_dataset_size exists (from your Streamlit slider/input), use it.
    # Otherwise, fall back to what the LLM suggested.
    final_size = gui_dataset_size if gui_dataset_size is not None else config.dataset_size
    
    print(f"   ✅ Configuration Generated: {config.project_name}")
    print(f"      Topic: {config.data_topic} | Size: {final_size} | Lang: {config.language}")

    # 6. Update State (using final_size instead of config.dataset_size)
    return {
        "project_name": config.project_name,
        "base_model": config.base_model,
        "data_topic": config.data_topic,
        "data_style": config.data_style,
        "dataset_size": final_size,  # <--- FIXED: GUI value is preserved
        "language": config.language,
        "status": "gathering_data",
        "messages": [AIMessage(content=f"Project initialized. I will find {final_size} examples regarding '{config.data_topic}'.")]
    }