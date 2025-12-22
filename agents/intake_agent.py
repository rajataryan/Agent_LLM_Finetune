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
If the user mentions a specific number of examples, set 'dataset_size' to that number. Otherwise default to 500.
If the user writes in a specific language (e.g., Hindi, Spanish), detect it and set the 'language' field.
"""

def intake_node(state: AgentState):
    print("--- 🧠 INTAKE AGENT STARTED (LOCAL) ---")
    
    # 1. Get User Input (Robust handling)
    messages = state.get("messages", [])
    if messages:
        user_input = messages[-1].content
    else:
        print("   ⚠️ No chat history found. Using 'user_goal' from state.")
        user_input = state.get("user_goal", "General LLM Fine-Tuning")

    print(f"   Analysing request: {user_input[:50]}...")

    # 2. Configure LLM for Structured Output
    # Llama 3.1 supports tool-calling features natively in LangChain
    structured_llm = llm.with_structured_output(ProjectConfiguration)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTAKE_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    # 3. Execute
    chain = prompt | structured_llm # the output of the prompt becomes the input of the structured LLM.
    config = chain.invoke({"input": user_input}) # the output of the chain is the final configuration.
    
    print(f"   ✅ Configuration Generated: {config.project_name}")
    print(f"      Topic: {config.data_topic} | Size: {config.dataset_size} | Lang: {config.language}")

    # 4. Update State
    return {
        "project_name": config.project_name,
        "base_model": config.base_model,
        "data_topic": config.data_topic,
        "data_style": config.data_style,
        "dataset_size": config.dataset_size,
        "language": config.language,
        "status": "gathering_data",
        "messages": [AIMessage(content=f"Project '{config.project_name}' initialized. I will find {config.dataset_size} examples regarding '{config.data_topic}'.")]
    }