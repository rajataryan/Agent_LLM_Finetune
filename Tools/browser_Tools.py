import os
from typing import List
from duckduckgo_search import DDGS
from langchain_community.document_loaders import BrowserbaseLoader

def search_web(query: str) -> List[str]:
    """
    Searches the web using DuckDuckGo.
    """
    clean_query = query.strip().replace('"', '').replace("'", "")
    print(f"🔎 SEARCHING (Backend: html): {clean_query}")
    
    urls = []
    
    try:
        with DDGS() as ddgs:
            # Enforce English results
            results = list(ddgs.text(clean_query, max_results=8, backend="html", region="us-en"))
            
            if not results:
                 results = list(ddgs.text(clean_query, max_results=8, backend="lite", region="us-en"))

            for r in results:
                link = r.get('href')
                if link and "baidu.com" not in link:
                    urls.append(link)
            
            # Limit to top 5
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
        print(f"   Reading: {url}...")
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
                combined_content += f"\n\n--- SOURCE: {url} ---\n{doc.page_content}"
            
        except Exception as e:
            print(f"   ❌ Scrape failed for {url}: {e}")
            
    return combined_content