# state.py
from typing import TypedDict, List, Annotated, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # NODE 1: Intake
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Project Config
    project_name: str
    base_model: str
    user_goal: str
    
    # Data Requirements
    data_topic: str
    data_style: str
    dataset_size: int
    language: str
    
    # NODE 2: Browser
    site_list: List[str]
    dataset_content: str
    
    # NODE 3: Data Logic
    training_file_path: str 
    generated_count: int  # <--- Tracks how much training data we have so far

    # NODE 4: Training
    job_id: str           # Modal Job ID (for tracking)
    training_status: str  # e.g., "submitted", "success", "failed"
    model_url: str        # The final Hugging Face link
    
    # Status Tracking
    status: str
    error: Optional[str]

    # finalizer Agent
    final_response: str