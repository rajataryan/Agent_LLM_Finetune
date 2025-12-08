import os
from dotenv import load_dotenv 

# 2. LOAD THE ENV FILE IMMEDIATELY
# This must happen BEFORE you initialize the ChatGoogleGenerativeAI model
load_dotenv() 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.messages import SystemMessage, HumanMessage 
from state import AgentState
from Tools.intake_tools import ProjectConfiguration 


#-----Model Setup-----
# Now this will work because the API Key is loaded into the environment
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(ProjectConfiguration)

#-----Prompt Setup-----
INTAKE_SYSTEM_PROMPT = """
You are an expert AI Solutions Architect. 
Your job is to listen to the user's request for a custom Fine-Tuned LLM 
and translate it into a technical configuration.

Analyze the user's request and fill out the configuration schema.
- Default to 'unsloth/llama-3-8b-bnb-4bit' if no model is specified.
- Infer the 'data_topic', 'data_style', and 'language' from context.
"""

# <----DEFINE THE TEMPLATE------
# You cannot use 'prompt_template' inside the function if you don't define it first!
prompt_template = ChatPromptTemplate.from_messages([
    ("system", INTAKE_SYSTEM_PROMPT),
    ("human", "{user_input}")
])

#-----Worker Node-----
def intake_agent(state: AgentState):
    """
    This function is the 'Worker' that LangGraph calls.
    It reads the State, does the work, and updates the State.
    """
    print("--- 🧠 INTAKE AGENT STARTED ---") # Added print so you can see it running

    # A. Get the last message from the user [-1] defines the last message
    last_user_message = state["messages"][-1].content

    # B. Format the prompt
    # Now this works because prompt_template is defined above
    formatted_prompt = prompt_template.format(user_input=last_user_message)

    # C. Call the LLM
    # invoke() expects the string/messages directly, not a dict {"messages": ...}
    result = structured_llm.invoke(formatted_prompt)

    # D. Update the Folder
    print(f"--- ✅ PLAN GENERATED: {result.project_name} ---")
    
    # From The Project Configuration tool
    return {
        "project_name": result.project_name,
        "base_model": result.base_model,
        # "user_goal": result.user_goal, # Can Add this Field in the Project Configuration tool
        "data_topic": result.data_topic,
        "data_style": result.data_style,
        "language": result.language,
        "status": "gathering_data",
        "messages": [result.confirmation_message]
    }

# --- OPTIONAL: TEST BLOCK ---
# If we want to run 'python -m agents.intake_agent' directly
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    # Mock data to simulate a user
    mock_state = {
        "messages": [HumanMessage(content="Make a bot that teaches me python in Hindi")]
    }
    
    output = intake_agent(mock_state)
    print("\nFINAL OUTPUT:", output)