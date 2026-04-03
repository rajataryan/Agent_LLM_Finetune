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
You are a Research Agent. 
Your goal is to find sources that contain DIALOGUE, SCRIPTS, or INTERVIEWS.
We need text that shows HOW people speak, not definitions of who they are.

Example:
Topic: "Strict Professor"
Bad Query: "Definition of strict professor"
Good Query: "Strict professor lecture script scene dialogue"
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
        
        # 🛡️ 1. PRE-EMPTIVE OVERRIDE 
        if any(k in user_goal for k in ["crush", "love", "shy", "romance", "flirt", "date", "kiss", "roleplay", "character"]):
            print("   ❤️ Romance/Roleplay Intent Detected -> SKIPPING SEARCH")
            print("   ✅ Forcing use of Curated Romance List")
            return {
                "site_list": ROMANCE_FALLBACK,
                "messages": [AIMessage(content="Using curated romance sources.")]
            }
            
        # 🛡️ 2. SMART SEARCH (The Upgrade)
        print("   Status: Starting Search.")
        
        # We ask Llama to be smarter about the query
        search_prompt = ChatPromptTemplate.from_messages([
            ("system", BROWSER_SYSTEM_PROMPT),
            ("human", "Generate 1 specific search query for: {topic}. Return ONLY the query string.")
        ])
        msg = search_prompt.format_messages(topic=topic, style=style)
        
        # Get the query from LLM
        base_query = llm.invoke(msg).content.strip().replace('"', '')
        
        # Force "dialogue" or "script" into the query if not present
        # This prevents getting dictionary definitions or Gmail help pages
        if "dialogue" not in base_query.lower() and "script" not in base_query.lower():
            base_query += " dialogue script examples"

        print(f"   🧠 Smart Query: {base_query}")
        
        # Run the search (The 'search_web' tool now handles the blacklisting of Google/Support sites)
        urls = search_web(base_query)
        
        # 🛡️ 3. FALLBACK LOGIC (If DuckDuckGo fails)
        if not urls:
            print("   ⚠️ Search blocked/empty. Using FALLBACK URLs.")
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

        # Limit to 5
        urls = urls[:5]
        
        return {
            "site_list": urls,
            "messages": [AIMessage(content=f"Found {len(urls)} sites.")]
        }

    # --- CHECKPOINT 2: SCRAPE ---
    elif existing_sites and not existing_content:
        print(f"   Status: Scraping {len(existing_sites)} sites.")
        
        # This uses your updated scrape_urls tool (with Browserbase)
        raw_content = scrape_urls(existing_sites)
        
        if not raw_content:
             print("   ⚠️ Scraping returned empty. Using Fallback content generation.")
             raw_content = "Generate synthetic data based on topic only."

        return {
            "dataset_content": raw_content,
            "status": "processing_raw_data",
            "messages": [AIMessage(content="I have read the material. Generating dataset...")]
        }

    else:
        return {"status": "processing_raw_data"}