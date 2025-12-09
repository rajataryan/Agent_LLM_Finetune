# tools/intake_tools.py
from pydantic import BaseModel, Field
from typing import Optional

class ProjectConfiguration(BaseModel):
    """
    The structured output we force the Intake Agent to generate.
    This acts as the 'Contract' between the User and the Backend.
    """
    project_name: str = Field(
        description="A short, slugified name for the project (no spaces). e.g., 'finance-bot-v1'"
    )
    base_model: str = Field(
        description="The huggingface model ID to use. Default to 'unsloth/llama-3-8b-bnb-4bit' unless requested otherwise.",
        default="unsloth/llama-3-8b-bnb-4bit"
    )
    data_topic: str = Field(
        description="The specific subject matter for the dataset. e.g., 'medical_diagnosis' or 'casual_slang'"
    )
    data_style: str = Field(
        description="The tone/style of the response. e.g., 'professional', 'pirate', 'gen-z'"
    )
    confirmation_message: str = Field(
        description="A short message to the user confirming you understood. e.g., 'Bet. I'm setting up a Llama-3 model to learn Gen Z slang.'"
    )
    language: str = Field(
        description="The primary language the bot should speak. Infer it from the user's prompt (e.g., if they write in Hindi, set this to 'Hindi').",
        default="English"
    )

# You can add a validation tool here if needed
def validate_project_name(name: str) -> bool:
    return " " not in name