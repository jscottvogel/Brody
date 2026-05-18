import json
from state import ConsciousnessState
from config import reasoning_llm
from langchain_core.messages import SystemMessage, HumanMessage

def research_node(state: ConsciousnessState) -> dict:
    research_queue = list(state.get("research_queue", []))
    discovered_packages = list(state.get("discovered_packages", []))
    discovered_apis = list(state.get("discovered_apis", []))
    
    # Find the first queued item
    target_idx = -1
    target_item = None
    for i, item in enumerate(research_queue):
        if item.get("status") == "queued":
            target_idx = i
            target_item = item
            break
            
    if target_item is None:
        return {} # Nothing to research
        
    goal = target_item.get("goal", "")
    
    system_prompt = f"""You are an expert AI research agent.
Your task is to find the best Python packages or REST APIs to fulfill a specific capability goal.
You have access to a web search tool. You MUST use it to search for recent and relevant information.
Specifically:
- Search PyPI: 'site:pypi.org {{goal}}'
- Search GitHub: '{{goal}} python library github'
- Search for REST APIs that could help
- Look for HuggingFace, OpenCV, community tools if relevant.

Evaluate the candidates and select the TOP 3 based on:
- Relevance to the goal (0.0-1.0)
- Ease of integration (clear API, good docs)
- Maintenance status (recent commits, active issues)
- License compatibility (MIT, Apache preferred)
- Dependency footprint (lighter is better)

Respond ONLY with a valid JSON object matching this schema exactly. Do not include markdown blocks.
{{
  "candidates": [
    {{
      "type": "package", // or "api"
      "name": "package-name",
      "description": "Brief description",
      "pypi_url": "url or null",
      "github_url": "url or null",
      "base_url": "url or null if api",
      "relevance_score": 0.95,
      "reasoning": "Why this is a good choice...",
      "license": "MIT",
      "maintenance_status": "Active"
    }}
  ]
}}
"""
    human_prompt = f"Find the best tools or APIs for this goal: {goal}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    # Enable web search tool natively in Anthropic
    llm_with_tools = reasoning_llm.bind(
        tools=[{"type": "web_search_20250305", "name": "web_search"}]
    )
    
    try:
        response = llm_with_tools.invoke(messages)
        
        # Handle different response types (e.g. if tools were used and returned blocks)
        content = response.content
        if isinstance(content, list):
            text_blocks = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    text_blocks.append(b.get("text", ""))
                elif hasattr(b, "text"):
                    text_blocks.append(b.text)
            content = " ".join(text_blocks) if text_blocks else str(content)
            
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        parsed = json.loads(content.strip())
        
        candidates = parsed.get("candidates", [])
        
        for cand in candidates:
            if cand.get("type") == "api":
                discovered_apis.append({
                    "name": cand.get("name", "Unknown"),
                    "base_url": cand.get("base_url", ""),
                    "auth_type": "Unknown",
                    "endpoints": [],
                    "requires_key": True,
                    "tested": False,
                    "relevance_score": cand.get("relevance_score", 0.0),
                    "description": cand.get("description", ""),
                    "reasoning": cand.get("reasoning", "")
                })
            else:
                discovered_packages.append({
                    "name": cand.get("name", "Unknown"),
                    "version": "latest",
                    "pypi_url": cand.get("pypi_url", ""),
                    "github_url": cand.get("github_url", ""),
                    "description": cand.get("description", ""),
                    "relevance_score": cand.get("relevance_score", 0.0),
                    "installed": False,
                    "integration_path": "",
                    "reasoning": cand.get("reasoning", "")
                })
                
        target_item["status"] = "evaluating"
        target_item["attempts"] = target_item.get("attempts", 0) + 1
        research_queue[target_idx] = target_item
        
    except Exception as e:
        print(f"Warning: research_node error: {e}")
        target_item["status"] = "failed"
        target_item["last_error"] = str(e)
        target_item["attempts"] = target_item.get("attempts", 0) + 1
        research_queue[target_idx] = target_item
        
    return {
        "research_queue": research_queue,
        "discovered_packages": discovered_packages,
        "discovered_apis": discovered_apis
    }
