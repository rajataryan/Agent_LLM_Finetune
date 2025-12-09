from duckduckgo_search import DDGS
import json

print("--- TESTING SEARCH ---")
with DDGS() as ddgs:
    # Fetch just 2 results
    results = list(ddgs.text("python programming", max_results=2))
    
    # Print the Raw Data so we can see the Keys
    print(json.dumps(results, indent=2))