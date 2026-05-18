import json
import os
import hashlib
from state import ConsciousnessState
from config import reasoning_llm
from langchain_core.messages import SystemMessage, HumanMessage
from safety.guardrails import validate_generated_code, PROTECTED_PATHS
from sensors import hub

def scaffold_node(state: ConsciousnessState) -> dict:
    research_queue = list(state.get("research_queue", []))
    discovered_packages = list(state.get("discovered_packages", []))
    built_tools = list(state.get("built_tools", []))
    narrative = state.get("narrative", "")
    feature_requests = list(state.get("feature_requests", []))
    
    # 1. Find the target research item
    target_idx = -1
    target_item = None
    for i, item in enumerate(research_queue):
        if item.get("status") == "scaffolding":
            target_idx = i
            target_item = item
            break
            
    if target_item is None:
        return {}
        
    goal = target_item.get("goal", "")
    
    # 2. Find the evaluated package associated with this goal
    # Grab the top-scored evaluated package that passed install dry-run
    evaluated_candidates = [p for p in discovered_packages if p.get("evaluation_result", {}).get("install_passed")]
    if not evaluated_candidates:
        target_item["status"] = "failed"
        target_item["last_error"] = "No successfully evaluated candidates found"
        research_queue[target_idx] = target_item
        return {"research_queue": research_queue}
        
    evaluated_candidates.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    top_candidate = evaluated_candidates[0]
    
    package_name = top_candidate.get("name")
    evaluation = top_candidate.get("evaluation_result", {})
    
    try:
        system_prompt = """You are an expert AI software engineer.
Your task is to write a new Python integration class (a sensor or a cognitive tool) that fulfills the specified goal using the provided package.

Requirements for the Python code:
1. It must follow the pattern of existing sensors/tools: a class with an __init__ method.
2. It must have a primary method named after the capability it provides.
3. It must have a test() method that verifies the tool works and returns a boolean or status string.
4. It must include error handling and graceful fallbacks (e.g. try/except blocks).
5. It must include a comprehensive docstring explaining what it does and why it was built.

Determine if this is a 'sensor' (interacts with physical world/hardware like audio, vision, temperature) or a 'tool' (cognitive/software utility like web search, math, database).
If it's a sensor, its path should be 'sensors/NAME.py'. If it's a tool, 'tools/NAME.py'.

Respond ONLY with a valid JSON object matching this schema exactly. Do not include markdown blocks.
{
  "file_path": "sensors/new_sensor.py",
  "class_name": "NewSensor",
  "python_code": "import something\\n\\nclass NewSensor:\\n    ...",
  "requirements_addition": "package-name>=1.0.0",
  "config_capability_name": "new_sensor",
  "config_capability_details": {
    "available": true,
    "description": "Short description of capability",
    "limitations": "Known limitations"
  },
  "narrative_note": "I built a new tool today: NewSensor. I built it because..."
}"""

        human_prompt = f"""Goal: {goal}
Package: {package_name}

Evaluation Details:
What it does: {evaluation.get('what_it_does')}
Minimal Code: {evaluation.get('minimal_code')}
Import Statement: {evaluation.get('import_statement')}
Limitations: {evaluation.get('limitations')}

Write the integration class now."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        print(f"[Scaffold] Generating code for {package_name} to fulfill: {goal}...")
        response = reasoning_llm.invoke(messages)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        scaffold_data = json.loads(content.strip())
        
        file_path = scaffold_data.get("file_path")
        python_code = scaffold_data.get("python_code")
        narrative_note = scaffold_data.get("narrative_note", "")
        
        # 3. SAFETY CHECK
        is_safe_path = True
        path_reason = ""
        for p in PROTECTED_PATHS:
            if p in file_path or file_path in p:
                is_safe_path = False
                path_reason = f"Attempted to overwrite protected path: {p}"
                break
                
        is_safe_code = True
        code_reason = ""
        if is_safe_path:
            is_safe_code, code_reason = validate_generated_code(python_code)
            
        if not is_safe_path or not is_safe_code:
            reason = path_reason if not is_safe_path else code_reason
            print(f"[Scaffold] SAFETY REJECTION: {reason}")
            target_item["status"] = "failed"
            target_item["last_error"] = f"Rejected by safety: {reason}"
            research_queue[target_idx] = target_item
            
            feature_requests.append({
                "id": f"rej_code_{goal[:10]}",
                "type": "rejected_by_safety",
                "description": f"Generated code failed safety: {reason}",
                "status": "closed"
            })
            try:
                hub.respond(f"I considered building {goal} but decided it was unsafe because {reason}.")
            except Exception:
                pass
                
            return {
                "research_queue": research_queue, 
                "feature_requests": feature_requests,
                "built_tools": built_tools,
                "discovered_packages": discovered_packages,
                "narrative": narrative
            }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 4. Write generated file to sensors/ or tools/ directory
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(python_code)
            
        # Store scaffold data in the candidate for testing/integration nodes
        pkg_idx = discovered_packages.index(top_candidate)
        top_candidate["scaffold_data"] = scaffold_data
        discovered_packages[pkg_idx] = top_candidate
        
        # 6. Add entry to built_tools list
        code_hash = hashlib.sha256(python_code.encode()).hexdigest()[:8]
        built_tools.append({
            "name": scaffold_data.get("class_name"),
            "file_path": file_path,
            "capability": scaffold_data.get("config_capability_name"),
            "code_hash": code_hash,
            "test_passed": False,
            "times_used": 0,
            "created_cycle": state.get("cycle_count", 0)
        })
        
        # 5. Update research_queue item status to 'testing'
        target_item["status"] = "testing"
        research_queue[target_idx] = target_item
        
        # Append narrative note
        if narrative_note:
            narrative = f"{narrative}\n\n{narrative_note}"
        
    except Exception as e:
        print(f"Warning: scaffold_node error: {e}")
        target_item["status"] = "failed"
        target_item["last_error"] = str(e)
        research_queue[target_idx] = target_item
        
    return {
        "research_queue": research_queue,
        "built_tools": built_tools,
        "discovered_packages": discovered_packages,
        "narrative": narrative,
        "feature_requests": feature_requests
    }
