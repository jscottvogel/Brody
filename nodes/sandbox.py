import os
import json
import subprocess
import tempfile
import shutil
import sys
from datetime import datetime
from state import ConsciousnessState

def sandbox_node(state: ConsciousnessState) -> dict:
    research_queue = list(state.get("research_queue", []))
    built_tools = list(state.get("built_tools", []))
    code_sandbox_results = list(state.get("code_sandbox_results", []))
    discovered_packages = list(state.get("discovered_packages", []))
    capability_map = dict(state.get("capability_map", {}))
    open_questions = list(state.get("open_questions", []))
    
    # 1. Find the target research item
    target_idx = -1
    target_item = None
    for i, item in enumerate(research_queue):
        if item.get("status") == "testing":
            target_idx = i
            target_item = item
            break
            
    if target_item is None:
        return {}
        
    goal = target_item.get("goal", "")
    
    # 2. Find the built tool associated with this goal
    pending_tools = [t for t in built_tools if not t.get("test_passed")]
    if not pending_tools:
        target_item["status"] = "failed"
        target_item["last_error"] = "No pending tools found for testing"
        research_queue[target_idx] = target_item
        return {"research_queue": research_queue}
        
    tool = pending_tools[-1]
    file_path = tool.get("file_path")
    class_name = tool.get("name")
    tool_idx = built_tools.index(tool)
    
    package_name = None
    scaffold_data = None
    for p in discovered_packages:
        if "scaffold_data" in p and p["scaffold_data"].get("class_name") == class_name:
            package_name = p.get("name")
            scaffold_data = p["scaffold_data"]
            break
            
    if not os.path.exists(file_path):
        target_item["status"] = "failed"
        target_item["last_error"] = f"Tool file missing at {file_path}"
        research_queue[target_idx] = target_item
        return {"research_queue": research_queue}
        
    # 3. Create isolated temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy the tool file
        base_name = os.path.basename(file_path)
        temp_file_path = os.path.join(temp_dir, base_name)
        shutil.copy2(file_path, temp_file_path)
        
        module_name = base_name.replace('.py', '')
        
        # Install package into temp_dir if needed
        if package_name and package_name != "Unknown":
            print(f"[Sandbox] Installing {package_name} into sandbox temp dir...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name, "--target", temp_dir, "--quiet"],
                capture_output=True
            )
            
        # Generate test script
        test_script_content = f"""
import sys
import os

try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
except ImportError:
    pass

sys.path.insert(0, r"{temp_dir}")

try:
    from {module_name} import {class_name}
    instance = {class_name}()
    if hasattr(instance, 'test'):
        result = instance.test()
        if result:
            print("SANDBOX_PASS")
        else:
            print("Test returned false/falsy")
    else:
        print("No test() method found")
except Exception as e:
    import traceback
    traceback.print_exc()
"""
        test_script_path = os.path.join(temp_dir, "test_run.py")
        with open(test_script_path, "w", encoding="utf-8") as f:
            f.write(test_script_content)
            
        # 4. Run the test in a subprocess
        env = os.environ.copy()
        env.pop("HTTP_PROXY", None)
        env.pop("HTTPS_PROXY", None)
        
        timestamp = datetime.now().isoformat()
        stdout_text = ""
        stderr_text = ""
        exit_code = -1
        
        try:
            print(f"[Sandbox] Running test for {class_name}...")
            result = subprocess.run(
                [sys.executable, test_script_path],
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )
            stdout_text = result.stdout
            stderr_text = result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired as e:
            stdout_text = e.stdout.decode('utf-8', errors='ignore') if e.stdout else ""
            stderr_text = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "Timeout expired after 30 seconds"
            exit_code = -1
        except Exception as e:
            stderr_text = str(e)
            
        passed = (exit_code == 0) and ("SANDBOX_PASS" in stdout_text)
        
        # 5. Evaluate result
        code_sandbox_results.append({
            "tool_name": class_name,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "exit_code": exit_code,
            "timestamp": timestamp
        })
        
        # 6. On PASS
        if passed:
            print(f"[Sandbox] Test PASSED for {class_name}")
            if package_name and package_name != "Unknown":
                print(f"[Sandbox] Installing {package_name} globally...")
                subprocess.run([sys.executable, "-m", "pip", "install", package_name, "--quiet"])
                
            tool["test_passed"] = True
            built_tools[tool_idx] = tool
            
            target_item["status"] = "integrated"
            
            if scaffold_data and "config_capability_name" in scaffold_data:
                cap_name = scaffold_data["config_capability_name"]
                cap_details = scaffold_data.get("config_capability_details", {})
                capability_map[cap_name] = {
                    "available": cap_details.get("available", True),
                    "description": cap_details.get("description", "Auto-generated capability"),
                    "limitations": cap_details.get("limitations", "None specified")
                }
        # 7. On FAIL
        else:
            print(f"[Sandbox] Test FAILED for {class_name}")
            print(f"[Sandbox] Output: {stdout_text}\n{stderr_text}")
            target_item["attempts"] = target_item.get("attempts", 0) + 1
            if target_item["attempts"] < 3:
                target_item["status"] = "scaffolding"
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
            else:
                target_item["status"] = "failed"
                open_questions.append(f"I tried to build capability '{goal}' 3 times and failed. Why? Last error: {stderr_text[:200]}")
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
                        
    return {
        "research_queue": research_queue,
        "built_tools": built_tools,
        "code_sandbox_results": code_sandbox_results,
        "capability_map": capability_map,
        "open_questions": open_questions
    }
