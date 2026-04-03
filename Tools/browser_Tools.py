import os
from typing import List
from duckduckgo_search import DDGS
from langchain_community.document_loaders import BrowserbaseLoader

# --- DOMAIN FILTERING ---
# These sites are "noise" for character/persona research. 
# We explicitly ban them from search results.
BLACKLIST_DOMAINS = [
    "support.google.com", "mail.google.com", "accounts.google.com",
    "microsoft.com", "support.apple.com", 
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com",
    "play.google.com", "apps.apple.com"
]

def search_web(query: str) -> List[str]:
    """
    Searches the web using DuckDuckGo, automatically filtering out support/help pages.
    """
    # 1. Clean the query
    clean_query = query.strip().replace('"', '').replace("'", "")
    
    # 2. Construct "Negative Constraints"
    # This adds " -site:support.google.com -site:microsoft.com ..." to the query
    exclusions = " ".join([f"-site:{d}" for d in BLACKLIST_DOMAINS])
    final_query = f"{clean_query} {exclusions}"

    print(f"🔎 SEARCHING (Smart Filter): {clean_query}")
    
    urls = []
    
    try:
        with DDGS() as ddgs:
            # We explicitly ask for "html" backend as per your config, 
            # but we request a few extra results (10) because we might filter some out manually.
            results = list(ddgs.text(final_query, max_results=10, backend="html", region="us-en"))
            
            # Fallback if html backend fails
            if not results:
                 print("   ⚠️ HTML backend empty, trying 'lite' backend...")
                 results = list(ddgs.text(final_query, max_results=10, backend="lite", region="us-en"))

            for r in results:
                link = r.get('href')
                
                # Double-check: Sometimes search engines ignore the "-site:" operator.
                # We manually skip if a blacklisted domain sneaked in.
                if link and "baidu.com" not in link:
                    if not any(blocked in link for blocked in BLACKLIST_DOMAINS):
                        urls.append(link)
            
            # Limit set to top 5 CLEAN results
            return urls[:5]
            
    except Exception as e:
        print(f"⚠️ Search CRASHED: {e}")
        return []

def scrape_urls(urls: List[str]) -> str:
    """
    Visits a list of URLs one-by-one using Browserbase Loader.
    """
    print(f"☁️ SCRAPING {len(urls)} SITES (BROWSERBASE)...")
    
    api_key = os.getenv("BROWSERBASE_API_KEY")
    project_id = os.getenv("BROWSERBASE_PROJECT_ID")
    
    if not api_key or not project_id:
        print("❌ Error: Missing BROWSERBASE_API_KEY or BROWSERBASE_PROJECT_ID in .env")
        return ""

    combined_content = ""
    
    for url in urls:
        print(f"   Reading: {url[:60]}...") # Truncate long URLs in logs
        try:
            # We load one by one so if one fails, the others still work
            loader = BrowserbaseLoader(
                urls=[url],
                text_content=True,
                api_key=api_key,
                project_id=project_id,
            )
            
            docs = loader.load()
            if docs:
                doc = docs[0]
                print(f"   ✅ Success!")
                # Add a separator so the LLM knows where one site ends and another begins
                combined_content += f"\n\n--- SOURCE: {url} ---\n{doc.page_content[:4000]}" # Limit chars per site
            
        except Exception as e:
            print(f"   ❌ Scrape failed for {url}: {e}")
            
    return combined_content