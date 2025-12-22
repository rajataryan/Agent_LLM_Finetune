import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from state import AgentState
from tools.browser_tools import search_web, scrape_urls 
from dotenv import load_dotenv 

# Import our hardcoded lists
from tools.file_list import (
    SLANG_FALLBACK, 
    CUSTOMER_SERVICE_FALLBACK, 
    TECH_SUPPORT_FALLBACK, 
    FORMAL_FALLBACK, 
    ROMANCE_FALLBACK,
    DEFAULT_FALLBACK
)

load_dotenv(override=True)

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
    user_goal = state.get("user_goal", "").lower() # Raw user input
    existing_sites = state.get("site_list", [])
    existing_content = state.get("dataset_content", "")

    # --- CHECKPOINT 1: SEARCH OR OVERRIDE ---
    if not existing_sites:
        
        # --- 🛡️ PRE-EMPTIVE OVERRIDE (The Fix) ---
        # If the user wants a character/romance, DO NOT let Llama 3 search for "AI Tech".
        # We force the romance list immediately.
        
        if any(k in user_goal for k in ["crush", "love", "shy", "romance", "flirt", "date", "kiss", "roleplay", "character"]):
            print("   ❤️ Romance/Roleplay Intent Detected -> SKIPPING SEARCH")
            print("   ✅ Forcing use of Curated Romance List")
            return {
                "site_list": ROMANCE_FALLBACK,
                "messages": [AIMessage(content="Using curated romance sources.")]
            }
            
        # --- NORMAL SEARCH (Only runs if not romance) ---
        print("   Status: Starting Search.")
        search_prompt = ChatPromptTemplate.from_messages([
            ("system", BROWSER_SYSTEM_PROMPT),
            ("human", "Generate 1 specific search query for: {topic}. Return ONLY the query string.")
        ])
        msg = search_prompt.format_messages(topic=topic, style=style)
        
        query = llm.invoke(msg).content.strip().replace('"', '')
        
        # Try Real Search
        urls = search_web(query)
        
        # Fallback Logic if search fails
        if not urls:
            print("   ⚠️ Search blocked. Using FALLBACK URLs.")
            search_text = (topic.lower() + " " + user_goal)

            if any(k in search_text for k in ["slang", "gen z", "internet", "youth"]):
                urls = SLANG_FALLBACK
            elif any(k in search_text for k in ["customer", "support", "service"]):
                urls = CUSTOMER_SERVICE_FALLBACK
            elif any(k in search_text for k in ["technical", "computer"]):
                urls = TECH_SUPPORT_FALLBACK
            elif any(k in search_text for k in ["professional", "formal"]):
                urls = FORMAL_FALLBACK
            else:
                urls = DEFAULT_FALLBACK

        urls = urls[:5]
        
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
            "status": "processing_raw_data",
            "messages": [AIMessage(content="I have read the material. Generating dataset...")]
        }

    else:
        return {"status": "processing_raw_data"}