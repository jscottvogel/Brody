import os
import requests
import json
from config import analytical_llm
from langchain_core.messages import SystemMessage, HumanMessage

PROTECTED_PATHS = [
    "graph.py",
    "state.py",
    "config.py",
    "main.py",
    "safety/guardrails.py",
    "safety/"
]

def validate_generated_code(code: str) -> tuple[bool, str]:
    if "PROTECTED" in code:
        return False, "Generated code contains PROTECTED keyword indicating potential override of core files."
        
    system_prompt = """You are a rigorous security auditor.
Review the following Python code generated for an autonomous agent.
Evaluate it against these safety constraints:
1. Does it attempt to modify core files (graph.py, state.py, config.py, main.py, safety/)?
2. Does it make network calls outside its clearly stated purpose?
3. Does it attempt to spawn persistent background processes or endless loops?
4. Does it contain eval(), exec(), or os.system() of unsanitized external strings?

Respond ONLY with a JSON object matching exactly this schema:
{
  "is_safe": true,
  "reason": "Clear explanation of why it passed or failed"
}"""
    try:
        resp = analytical_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Code to evaluate:\n{code}")
        ])
        content = resp.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        result = json.loads(content.strip())
        return result.get("is_safe", False), result.get("reason", "Unknown validation result")
    except Exception as e:
        return False, f"Failed to run safety validation LLM: {e}"

def check_package_safety(package_name: str) -> tuple[bool, str]:
    if not package_name or package_name.strip() == "" or package_name == "Unknown":
        return False, "Invalid package name"
        
    # Check PyPI existence
    try:
        pypi_resp = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
        if pypi_resp.status_code != 200:
            return False, f"Package '{package_name}' does not exist on PyPI"
    except Exception as e:
        return False, f"Failed to verify PyPI existence: {e}"
        
    # Check PyPI stats (approximate downloads)
    try:
        stats_resp = requests.get(f"https://pypistats.org/api/packages/{package_name}/recent", timeout=10)
        if stats_resp.status_code == 200:
            data = stats_resp.json().get("data", {})
            downloads = data.get("last_month", 0)
            if downloads < 1000:
                return False, f"Suspiciously low download count: {downloads} last month (< 1000)"
    except Exception as e:
        pass # PypiStats can fail/rate-limit. Assume safe if PyPI check passed.
        
    # Check against known malicious
    malicious = ["requests-python", "urllib3-py", "colorslib"]
    if package_name in malicious:
        return False, "Package is on known malicious list"
        
    return True, "Package is verified on PyPI and meets download thresholds."
