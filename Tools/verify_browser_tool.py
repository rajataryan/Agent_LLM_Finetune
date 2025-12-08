import os
import sys
from dotenv import load_dotenv

# Ensure we can import the tool
sys.path.append(os.getcwd())
load_dotenv(override=True)

try:
    from Tools.browser_Tools import browser_base_tool
    
    # Test query
    query = "LangChain agents tutorial"
    print(f"Testing browser_base_tool with query: '{query}'")
    
    # Since it's a tool, we might need to invoke it properly or call its func if exposed, 
    # but based on definition: @tool def browser_base_tool(query: str) -> str:
    # We can call it directly as a function for testing logic, or via .run if it's a structured tool
    
    # Direct function call if untyped, or .run call
    try:
        result = browser_base_tool.invoke(query)
    except AttributeError:
        # If invoke not available (depending on langchain version/setup), try calling direct
        result = browser_base_tool(query)
        
    print("\n--- Result Snippet (first 500 chars) ---")
    print(result[:500])
    
    if "Error" in result and "Scraping" not in result:
        print("Test FAILED with error.")
        sys.exit(1)
    else:
        print("Test PASSED (Data returned).")

except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Execution failed: {e}")
    sys.exit(1)
