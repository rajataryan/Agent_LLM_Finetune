# DataGenerator.py
from typing import Annotated, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import Image, display
import gradio as gr
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import os
from langchain_openai import ChatOpenAI
from typing import TypedDict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agenttools.dataGenAgent_tools import dataGenerator_tools  # your helper utilities

load_dotenv(override=True)

# Config / constants
SAVE_DATAFILE = "trainingData.json"
INSTRUCTIONS = f"""You are an Agent that generates sample data for fine-tuning LLMs.
The dataset must be valid JSON (JSONL or an array of objects) and saved locally as: {SAVE_DATAFILE}
Follow the user's spec strictly. You may call helper tools and the web if needed.
"""

# --- Node implementations (functions that operate on a shared state dict) ---

def intake_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read requirement spec (from UI or state) and normalize it.
    Expects: state['user_spec'] set by caller (or gradio UI).
    Produces: state['task_spec'] which is a normalized dict describing dataset needs.
    """
    raw = state.get("user_spec", "").strip()
    if not raw:
        raise ValueError("No user spec provided to intake_agent.")
    # Simple normalization: you can replace this with a call to an LLM agent to parse the spec.
    task_spec = dataGenerator_tools.parse_user_spec(raw)
    state["task_spec"] = task_spec
    add_messages(state, f"[intake_agent] parsed task_spec: {task_spec}")
    return state


def generate_dataset_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates dataset examples according to the task_spec.
    Delegates heavy LLM/generation logic to dataGenerator_tools so you can iterate there.
    Produces: state['dataset_path'] pointing to local file, and state['dataset_obj'] (in-memory).
    """
    spec = state.get("task_spec")
    if spec is None:
        raise ValueError("generate_dataset_agent requires state['task_spec']")
    # The tool should return a list of dicts (or JSONL string) and optionally save a local file.
    dataset_obj = dataGenerator_tools.create_dataset_from_spec(spec, n_examples=spec.get("dataset_size", 50))
    # Ensure it's a JSON-serializable list/dict
    dataset_path = os.path.abspath(SAVE_DATAFILE)
    dataGenerator_tools.save_json(dataset_obj, dataset_path)
    state["dataset_obj"] = dataset_obj
    state["dataset_path"] = dataset_path
    add_messages(state, f"[generate_dataset_agent] saved dataset to {dataset_path}")
    return state


def evaluate_dataset_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run quick automatic checks: schema, duplicates, length distribution, toxicity heuristics, etc.
    If issues are found, attach them to state['eval_report'] and mark state['fix_required'].
    """
    dataset = state.get("dataset_obj")
    if dataset is None:
        raise ValueError("evaluate_dataset_agent requires state['dataset_obj']")
    report = dataGenerator_tools.evaluate_dataset_quality(dataset)
    state["eval_report"] = report
    # Decide if dataset needs fixes (boolean)
    state["fix_required"] = report.get("quality_score", 1.0) < 0.85 or bool(report.get("issues"))
    add_messages(state, f"[evaluate_dataset_agent] quality_score={report.get('quality_score')}, issues={report.get('issues')}")
    return state


def finalize_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final packaging: create a manifest and return result summary for UI.
    """
    dataset_path = state.get("dataset_path")
    report = state.get("eval_report", {})
    manifest = {
        "dataset_path": dataset_path,
        "num_examples": len(state.get("dataset_obj", [])),
        "eval": report,
    }
    manifest_path = os.path.abspath("dataset_manifest.json")
    dataGenerator_tools.save_json(manifest, manifest_path)
    state["manifest_path"] = manifest_path
    add_messages(state, f"[finalize_agent] manifest saved to {manifest_path}")
    return state
