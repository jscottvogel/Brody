import json
import requests
import time
import os
import hashlib
from state import ConsciousnessState
from config import reasoning_llm, analytical_llm
from sensors import hub
from langchain_core.messages import SystemMessage, HumanMessage

def api_discover_node(state: ConsciousnessState) -> dict:
    research_queue = list(state.get("research_queue", []))
    discovered_apis = list(state.get("discovered_apis", []))
    feature_requests = list(state.get("feature_requests", []))
    built_tools = list(state.get("built_tools", []))
    
    target_idx = -1
    target_item = None
    for i, item in enumerate(research_queue):
        if item.get("status") == "api_discovery":
            target_idx = i
            target_item = item
            break
            
    if target_item is None:
        return {}
        
    goal = target_item.get("goal", "")
    
    system_prompt = f"""You are an expert API researcher.
Your task is to find the best free or open REST APIs to fulfill a specific capability goal.
You have access to a web search tool. You MUST use it to search for recent and relevant information.
Search queries to try:
- 'free API {{goal}} no auth'
- 'open API {{goal}} json'
- Check the public-apis GitHub list, RapidAPI free tier, government open data, and HuggingFace inference API.

Respond ONLY with a valid JSON object matching this schema exactly. Do not include markdown blocks.
{{
  "apis": [
    {{
      "name": "api-name",
      "base_url": "https://api.example.com/v1",
      "test_endpoint": "https://api.example.com/v1/test",
      "requires_key": false,
      "auth_type": "none", 
      "env_var_name": null,
      "description": "Brief description"
    }}
  ]
}}"""

    human_prompt = f"Find the best REST APIs for this goal: {goal}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    llm_with_tools = reasoning_llm.bind(
        tools=[{"type": "web_search_20250305", "name": "web_search"}]
    )
    
    try:
        print(f"[API Discover] Searching for APIs for goal: {goal}...")
        response = llm_with_tools.invoke(messages)
        
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
        apis = parsed.get("apis", [])
        
        viable_apis = []
        
        for api in apis:
            requires_key = api.get("requires_key", False)
            env_var_name = api.get("env_var_name")
            
            has_key = False
            if requires_key and env_var_name:
                if os.getenv(env_var_name):
                    has_key = True
                else:
                    feature_requests.append({
                        "id": f"req_api_{env_var_name}",
                        "title": f"API Key Needed: {api.get('name')}",
                        "type": "api_key_needed",
                        "description": f"Need API key in environment variable {env_var_name} to fulfill {goal}",
                        "status": "pending",
                        "env_var_name": env_var_name
                    })
                    hub.respond(f"I found a useful API but need a key: {api.get('name')}.")
            
            test_endpoint = api.get("test_endpoint")
            if test_endpoint and (not requires_key or has_key):
                try:
                    start_time = time.time()
                    headers = {}
                    if requires_key and has_key and api.get("auth_type") == "header":
                        headers["Authorization"] = f"Bearer {os.getenv(env_var_name)}"
                        
                    res = requests.get(test_endpoint, headers=headers, timeout=10)
                    elapsed = time.time() - start_time
                    
                    if res.status_code == 200:
                        try:
                            json_data = res.json()
                            api["probe_result"] = {
                                "status_code": 200,
                                "response_time_sec": elapsed,
                                "structure": str(type(json_data).__name__)
                            }
                            api["tested"] = True
                            viable_apis.append(api)
                            print(f"[API Discover] Probe PASSED for {api.get('name')}")
                        except ValueError:
                            print(f"[API Discover] Probe FAILED for {api.get('name')}: Not JSON")
                    else:
                        print(f"[API Discover] Probe FAILED for {api.get('name')}: HTTP {res.status_code}")
                except Exception as probe_err:
                    print(f"[API Discover] Probe ERROR for {api.get('name')}: {probe_err}")
            elif requires_key and not has_key:
                api["tested"] = False
                viable_apis.append(api)

        for api in viable_apis:
            wrapper_system = """You are an expert Python developer.
Generate a wrapper class for the provided REST API.
Requirements:
1. Thin Python class with one method per relevant endpoint.
2. Handles auth (API key from env var if needed).
3. Parses response into a clean dict for robot reasoning.
4. Follows the same pattern as sensors/ classes (e.g. __init__, test method).

Respond ONLY with a valid JSON object matching this schema exactly. Do not include markdown blocks.
{
  "wrapper_code": "import requests\\nimport os\\n\\nclass ApiWrapper:\\n...",
  "class_name": "ApiWrapper"
}"""
            wrapper_human = f"API Details:\n{json.dumps(api, indent=2)}\nGoal: {goal}"
            
            try:
                wrap_resp = analytical_llm.invoke([
                    SystemMessage(content=wrapper_system),
                    HumanMessage(content=wrapper_human)
                ])
                wrap_content = wrap_resp.content.strip()
                if wrap_content.startswith("```json"):
                    wrap_content = wrap_content[7:-3]
                elif wrap_content.startswith("```"):
                    wrap_content = wrap_content[3:-3]
                    
                wrap_data = json.loads(wrap_content.strip())
                api["wrapper_class"] = wrap_data.get("wrapper_code")
                api["class_name"] = wrap_data.get("class_name")
                
                # Write to tools/ directory
                file_path = f"tools/{api['class_name'].lower()}.py"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(api["wrapper_class"])
                
                # Add to built_tools so sandbox can test it
                code_hash = hashlib.sha256(api["wrapper_class"].encode()).hexdigest()[:8]
                built_tools.append({
                    "name": api["class_name"],
                    "file_path": file_path,
                    "capability": api["name"].lower().replace(' ', '_'),
                    "code_hash": code_hash,
                    "test_passed": False,
                    "times_used": 0,
                    "created_cycle": state.get("cycle_count", 0)
                })
                
                discovered_apis.append(api)
            except Exception as wrap_e:
                print(f"[API Discover] Warning: Failed to generate wrapper for {api.get('name')}: {wrap_e}")
                
        if viable_apis:
            target_item["status"] = "testing" # Ready for sandbox test
        else:
            target_item["attempts"] = target_item.get("attempts", 0) + 1
            if target_item["attempts"] >= 3:
                target_item["status"] = "failed"
                target_item["last_error"] = "No viable APIs found after 3 attempts"
                
        research_queue[target_idx] = target_item
                
    except Exception as e:
        print(f"Warning: api_discover_node error: {e}")
        target_item["status"] = "failed"
        target_item["last_error"] = str(e)
        research_queue[target_idx] = target_item
        
    return {
        "research_queue": research_queue,
        "discovered_apis": discovered_apis,
        "feature_requests": feature_requests,
        "built_tools": built_tools
    }
