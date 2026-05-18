import json
import subprocess
import requests
import sys
from state import ConsciousnessState
from config import analytical_llm
from langchain_core.messages import SystemMessage, HumanMessage
from safety.guardrails import check_package_safety
from sensors import hub

def evaluate_node(state: ConsciousnessState) -> dict:
    research_queue = list(state.get("research_queue", []))
    discovered_packages = list(state.get("discovered_packages", []))
    feature_requests = list(state.get("feature_requests", []))
    
    # 1. Find the target research item
    target_idx = -1
    target_item = None
    for i, item in enumerate(research_queue):
        if item.get("status") == "evaluating":
            target_idx = i
            target_item = item
            break
            
    if target_item is None:
        return {}
        
    # 2. Find the top candidate package that hasn't been evaluated yet
    candidates = [p for p in discovered_packages if "evaluation_result" not in p]
    if not candidates:
        target_item["status"] = "failed"
        target_item["last_error"] = "No viable, installable package candidates found"
        research_queue[target_idx] = target_item
        return {"research_queue": research_queue}
        
    # Sort by relevance score
    candidates.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    top_candidate = candidates[0]
    package_name = top_candidate.get("name")
    
    if not package_name or package_name == "Unknown":
        top_candidate["evaluation_result"] = {"error": "Invalid package name", "install_passed": False}
        pkg_idx = discovered_packages.index(top_candidate)
        discovered_packages[pkg_idx] = top_candidate
        return {"discovered_packages": discovered_packages}
    
    pkg_idx = discovered_packages.index(top_candidate)
    
    # 3. SAFETY CHECK
    is_safe, reason = check_package_safety(package_name)
    if not is_safe:
        print(f"[Evaluate] SAFETY REJECTION for {package_name}: {reason}")
        top_candidate["evaluation_result"] = {"error": f"Rejected by safety: {reason}", "install_passed": False}
        discovered_packages[pkg_idx] = top_candidate
        
        target_item["status"] = "failed"
        target_item["last_error"] = f"Rejected by safety: {reason}"
        research_queue[target_idx] = target_item
        
        feature_requests.append({
            "id": f"rej_pkg_{package_name}",
            "type": "rejected_by_safety",
            "description": f"Package {package_name} failed safety: {reason}",
            "status": "closed"
        })
        try:
            hub.respond(f"I considered using {package_name} but decided it was unsafe. {reason}")
        except Exception:
            pass
            
        return {
            "research_queue": research_queue, 
            "discovered_packages": discovered_packages,
            "feature_requests": feature_requests
        }
    
    try:
        # 4. Fetch README
        readme_content = ""
        pypi_resp = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
        if pypi_resp.status_code == 200:
            readme_content = pypi_resp.json().get("info", {}).get("description", "")
            
        if not readme_content or len(readme_content) < 100:
            readme_content = f"Failed to fetch detailed README for {package_name}. Description: {top_candidate.get('description', '')}"
            
        # 4. Pass to analytical_llm
        system_prompt = """You are an expert software evaluator.
Read the provided package README/documentation and answer the following questions.
Respond ONLY with a valid JSON object matching this schema exactly. Do not include markdown blocks.
{
  "what_it_does": "What does this library actually do?",
  "minimal_code": "What is the minimal code to initialize and use it? (Provide 5-10 lines of Python)",
  "import_statement": "The exact import statement needed",
  "capability_description": "A one-line description of what capability it adds",
  "limitations": "What are the known limitations or failure modes?",
  "compatibility": "What Python version and OS compatibility does it have?",
  "security_concerns": "Are there any security concerns?"
}"""

        human_prompt = f"Package: {package_name}\n\nREADME/Docs:\n{readme_content[:15000]}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = analytical_llm.invoke(messages)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        evaluation = json.loads(content.strip())
        
        # 5. Attempt dry-run install
        print(f"[Evaluate] Running dry-run pip install for {package_name}...")
        dry_run_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name, "--dry-run", "--quiet"],
            capture_output=True,
            text=True
        )
        
        install_passed = (dry_run_result.returncode == 0)
        evaluation["install_passed"] = install_passed
        evaluation["install_stderr"] = dry_run_result.stderr
        
        top_candidate["evaluation_result"] = evaluation
        
        if install_passed:
            target_item["status"] = "scaffolding"
            
        discovered_packages[pkg_idx] = top_candidate
        research_queue[target_idx] = target_item
        
    except Exception as e:
        print(f"Warning: evaluate_node error: {e}")
        top_candidate["evaluation_result"] = {"error": str(e), "install_passed": False}
        discovered_packages[pkg_idx] = top_candidate
        
    return {
        "research_queue": research_queue,
        "discovered_packages": discovered_packages,
        "feature_requests": feature_requests
    }
