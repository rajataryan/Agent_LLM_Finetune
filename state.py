# state.py
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Chat History
    messages: List[BaseMessage]
    
    # The Blueprints (Set by Intake Agent)
    project_name: str           # e.g., "genz-slang-bot"
    base_model: str             # e.g., "unsloth/llama-3-8b-bnb-4bit"
    user_goal: str              # e.g., "Make the LLM talk like a Gen Z teenager"
    
    # Data Requirements (For Browser Agent)
    data_topic: str             # e.g., "slang_conversation"
    data_style: str             # e.g., "informal, gen-z, sarcastic"

    # Language Model Parameters
    language : str              # e.g., "English"
    
    # Status Flags
    status: str                 # "intake", "gathering_data", "training", "done"
    error: Optional[str]        #If something breaks, we write the error here so the 'debugger' node can see it.