import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from state import AgentState
from Tools.browser_tools import search_web, scrape_urls 
from dotenv import load_dotenv 

load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

BROWSER_SYSTEM_PROMPT = """
You are a Research Agent. Your job is to find raw text data.
Topic: {topic}
Style: {style}
"""

def browser_node(state: AgentState):
    print("--- 🌐 BROWSER AGENT STARTED ---")
    
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
        query = llm.invoke(msg).content
        
        urls = search_web(query)
        
        # --- FALLBACK MECHANISM ---
        if not urls:
            print("   ⚠️ Search blocked. Using FAST FALLBACK URLs.")
            if "slang" in topic.lower():
                # Wiki is much faster than Dictionary.com
                urls = [
                    "https://simple.wikipedia.org/wiki/Slang", 
                    "https://en.wikipedia.org/wiki/Internet_slang"
                ]
            else:
                urls = ["https://en.wikipedia.org/wiki/Python_(programming_language)"]
        # ---------------------------
        
        return {
            "site_list": urls,
            "messages": [AIMessage(content=f"Found {len(urls)} sites.")]
        }

    # --- CHECKPOINT 2: SCRAPE ---
    elif existing_sites and not existing_content:
        print(f"   Status: Scraping {len(existing_sites)} sites.")
        
        raw_content = scrape_urls(existing_sites)
        
        return {
            "dataset_content": raw_content,
            "status": "processing_data",
            "messages": [AIMessage(content="I have read the material. Generating dataset...")]
        }

    # --- CHECKPOINT 3: DONE ---
    else:
        return {"status": "processing_data"}

# --- TEST BLOCK ---
if __name__ == "__main__":
    mock_state = {
        "data_topic": "Gen Z Slang",
        "data_style": "Informal",
        "site_list": [], 
        "dataset_content": ""
    }
    
    print("\n--- RUN 1 (SEARCH) ---")
    result1 = browser_node(mock_state)
    print(result1)
    
    print("\n--- RUN 2 (SCRAPE) ---")
    mock_state["site_list"] = result1["site_list"]
    result2 = browser_node(mock_state)
    # Don't print the whole content, it's too big
    print(f"FINAL CONTENT LENGTH: {len(result2.get('dataset_content', ''))} characters")