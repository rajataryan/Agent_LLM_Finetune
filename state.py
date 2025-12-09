# state.py
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated

class AgentState(TypedDict):
    # NODE 1
    # Chat History
    messages: Annotated[List[BaseMessage], add_messages]
    
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

    # NODE 2
    # Stores Raw URLs found by the search tool
    site_list: List[str]
    dataset_content: str # Stores the massive string of scraped text

