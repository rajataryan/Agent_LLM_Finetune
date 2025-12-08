from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import Image, display
import gradio as gr
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import os
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from load_dotenv import load_dotenv

load_dotenv(override=True)

llm1 =ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
llm2 =ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

class State(TypedDict):
    pass
    
#graph_builder = StateGraph(State)
