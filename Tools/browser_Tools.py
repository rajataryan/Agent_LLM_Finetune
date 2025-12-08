from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool, Tool
import requests
from bs4 import BeautifulSoup
import re

serper = GoogleSerperAPIWrapper()

def search_func(query: str):
    return serper.run(query)

tool_search = Tool(
    name="search",
    func=search_func,
    description="Use this tool to look up anything you want in the internet"
)

@tool
def browser_base_tool(query: str) -> str:
    """
    Search the web for a query and scrape the content of the top results to build a dataset.
    Returns a string containing the aggregated content from the search results.
    Args:
        query: The search topic or query to find data for.
    """
    print(f"Searching for: {query}")
    try:
        # 1. Search for the query
        search_results = serper.results(query)
        organic_results = search_results.get("organic", [])
        
        gathered_data = []
        
        # 2. Process top 3 results
        for result in organic_results[:3]:
            link = result.get("link")
            title = result.get("title")
            snippet = result.get("snippet", "")
            
            if not link:
                continue
                
            print(f"Scraping: {link}")
            try:
                # 3. Request the page
                response = requests.get(link, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # 4. Extract text (simple extraction)
                    # Remove scripts and styles
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.decompose()
                        
                    text = soup.get_text(separator="\n")
                    
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    # Limit length per article
                    text = text[:4000] 
                    
                    gathered_data.append(f"Source: {title} ({link})\nSnippet: {snippet}\nContent:\n{text}\n{'-'*40}")
                else:
                    gathered_data.append(f"Failed to fetch {link} (Status: {response.status_code})")
            except Exception as e:
                gathered_data.append(f"Error scraping {link}: {str(e)}")
                
        if not gathered_data:
            return "No data found."
            
        return "\n\n".join(gathered_data)
        
    except Exception as e:
        return f"Error in browser_base_tool: {str(e)}"

browser_tools = [tool_search, browser_base_tool]