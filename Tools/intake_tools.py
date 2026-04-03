# tools/intake_tools.py
from pydantic import BaseModel, Field
from typing import Optional

class ProjectConfiguration(BaseModel):
    """
    Defines the structure for a fine-tuning project.
    """
    project_name: str = Field(
        description="A short, slugified name for the project (e.g., 'math-tutor-v1')."
    )
    base_model: str = Field(
        description="The base model ID.",
        default="unsloth/llama-3-8b-bnb-4bit"
    )
    data_topic: str = Field(
        description="The specific subject matter or persona (e.g., 'Physics', 'Customer Support')."
    )
    data_style: str = Field(
        description="The tone/style of the response (e.g., 'Formal', 'Sarcastic', 'Concise')."
    )
    dataset_size: int = Field(
        description="The EXACT number of samples requested by the user.",
        default=100  # Neutral default (was 500)
    )
    language: str = Field(
        description="The primary language for the dataset.",
        default="English"
    )

# You can add a validation tool here if needed
def validate_project_name(name: str) -> bool:
    return " " not in name