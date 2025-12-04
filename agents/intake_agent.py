from langgraph.graph.message import add_messages
from tools.data_generator_tools import parse_user_spec

def intake_agent(state):
    """
    Reads the user's text requirement and converts it into a structured task spec.
    """
    user_input = state.get("user_spec", "").strip()

    if not user_input:
        raise ValueError("Missing user_spec in state.")

    task_spec = parse_user_spec(user_input)
    state["task_spec"] = task_spec

    add_messages(state, f"[intake_agent] Parsed task spec: {task_spec}")
    return state