import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from state import AgentState
# Import tool functionality only
from tools.browser_tools import search_web, scrape_urls 
from dotenv import load_dotenv 

load_dotenv(override=True)

# We use Llama 3.2 (Small) here because generating a search query is easy and we want speed
llm = ChatOllama(model="llama3.2", temperature=0)

BROWSER_SYSTEM_PROMPT = """
You are a Research Agent. Your job is to find raw text data.
Topic: {topic}
Style: {style}
"""

def browser_node(state: AgentState):
    print("--- 🌐 BROWSER AGENT STARTED (LOCAL) ---")
    
    topic = state.get("data_topic", "General")
    style = state.get("data_style", "Normal")
    existing_sites = state.get("site_list", [])
    existing_content = state.get("dataset_content", "")

    # --- CHECKPOINT 1: SEARCH ---
    if not existing_sites:
        print("   Status: Starting Search.")
        
        search_prompt = ChatPromptTemplate.from_messages([
            ("system", BROWSER_SYSTEM_PROMPT),
            ("human", "Generate 1 specific search query for: {topic}. Return ONLY the query string.")
        ])
        msg = search_prompt.format_messages(topic=topic, style=style)
        
        # Cleanup response just in case Llama adds quotes
        query = llm.invoke(msg).content.strip().replace('"', '')
        
        # Call the Tool
        urls = search_web(query)
        
        # Fallback Logic
        if not urls:
            print("   ⚠️ Search blocked. Using FAST FALLBACK URLs.")
            if "slang" in topic.lower():
                urls = [
                    "https://simple.wikipedia.org/wiki/Slang", 
                    "https://en.wikipedia.org/wiki/Internet_slang"
                ]
            else:
                urls = ["https://en.wikipedia.org/wiki/Python_(programming_language)"]
        
        return {
            "site_list": urls,
            "messages": [AIMessage(content=f"Found {len(urls)} sites.")]
        }

    # --- CHECKPOINT 2: SCRAPE ---
    elif existing_sites and not existing_content:
        print(f"   Status: Scraping {len(existing_sites)} sites.")
        
        # Call the Tool
        raw_content = scrape_urls(existing_sites)
        
        return {
            "dataset_content": raw_content,
            "status": "processing_data",
            "messages": [AIMessage(content="I have read the material. Generating dataset...")]
        }

    else:
        return {"status": "processing_data"}