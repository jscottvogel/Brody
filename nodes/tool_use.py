import sys
import os
import importlib
import traceback
import json
import inspect
from datetime import datetime
from state import ConsciousnessState
from config import analytical_llm
from langchain_core.messages import SystemMessage, HumanMessage

def tool_use_node(state: ConsciousnessState) -> dict:
    built_tools = list(state.get("built_tools", []))
    sensor_input = state.get("sensor_input", "")
    open_questions = list(state.get("open_questions", []))
    curiosity_queue = list(state.get("curiosity_queue", []))
    code_sandbox_results = list(state.get("code_sandbox_results", []))
    
    # 1. Identify working tools
    working_tools = [t for t in built_tools if t.get("test_passed", False)]
    if not working_tools:
        return {} # No tools available to use
        
    # 2. Decide which tools are relevant
    system_prompt = """You are the tool-selection cognitive sub-module.
Review the available tools and the current open questions and curiosity queue.
Decide which tools (if any) should be invoked this cycle to help answer the questions.
Respond ONLY with a valid JSON array containing the exact names of the tools to use, e.g. ["WeatherSensor", "MathTool"].
If none are relevant, return []."""

    tools_str = "\n".join([f"- {t['name']} (Capability: {t.get('capability')})" for t in working_tools])
    human_prompt = f"""Available Tools:\n{tools_str}\n
Open Questions: {open_questions}
Curiosity Queue: {curiosity_queue}

Select tools to run:"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    selected_tool_names = []
    try:
        response = analytical_llm.invoke(messages)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        selected_tool_names = json.loads(content.strip())
        if not isinstance(selected_tool_names, list):
            selected_tool_names = []
    except Exception as e:
        print(f"[Tool Use] Failed to select tools: {e}")
        return {}
        
    if not selected_tool_names:
        return {}
        
    # Add root to path so importlib can find tools/ and sensors/ modules
    if "." not in sys.path:
        sys.path.insert(0, ".")
        
    tool_outputs = []
        
    # 3. Import and use tools
    for tool_name in selected_tool_names:
        # Find tool definition
        tool_def = next((t for t in working_tools if t["name"] == tool_name), None)
        if not tool_def:
            continue
            
        tool_idx = built_tools.index(tool_def)
        file_path = tool_def.get("file_path", "")
        
        # Derive module name from file path, e.g., tools/my_tool.py -> tools.my_tool
        if file_path.endswith(".py"):
            module_name = file_path[:-3].replace(os.sep, ".").replace("/", ".")
        else:
            module_name = f"tools.{tool_name.lower()}"
            
        try:
            print(f"[Tool Use] Invoking tool: {tool_name}...")
            # Dynamically import module
            module = importlib.import_module(module_name)
            importlib.reload(module)
            
            # Instantiate class
            class_obj = getattr(module, tool_name)
            instance = class_obj()
            
            # Find the primary method. Prioritize 'run', then the capability name, then the first public method
            method_to_call = None
            if hasattr(instance, "run"):
                method_to_call = getattr(instance, "run")
            else:
                cap = tool_def.get("capability")
                if cap and hasattr(instance, cap):
                    method_to_call = getattr(instance, cap)
                else:
                    for attr_name in dir(instance):
                        if not attr_name.startswith("_") and attr_name != "test":
                            attr = getattr(instance, attr_name)
                            if callable(attr):
                                method_to_call = attr
                                break
                                
            if not method_to_call:
                raise ValueError(f"Could not find an executable method on {tool_name}")
                
            # Execute dynamically based on signature
            sig = inspect.signature(method_to_call)
            if "context" in sig.parameters:
                result = method_to_call(context=sensor_input)
            elif len(sig.parameters) > 0:
                result = method_to_call(sensor_input)
            else:
                result = method_to_call()
                
            tool_outputs.append(f"[{tool_name} Output]: {result}")
            
            # 5. Track usage
            tool_def["times_used"] = tool_def.get("times_used", 0) + 1
            built_tools[tool_idx] = tool_def
            
        except Exception as e:
            err_trace = traceback.format_exc()
            print(f"[Tool Use] Tool {tool_name} failed during live use: {e}")
            
            # 6. Log error and mark broken
            code_sandbox_results.append({
                "tool_name": tool_name,
                "stdout": "",
                "stderr": err_trace,
                "exit_code": 1,
                "timestamp": datetime.now().isoformat(),
                "context": "live_use"
            })
            
            tool_def["test_passed"] = False
            built_tools[tool_idx] = tool_def
            
            open_questions.append(f"My {tool_name} tool failed during live use. Should I rebuild it? Error: {str(e)}")
            
    # 4. Pass tool output into sensor input
    if tool_outputs:
        sensor_input_str = str(sensor_input) if sensor_input else ""
        if sensor_input_str:
            sensor_input_str += "\n\n"
        sensor_input_str += "=== Tool execution results ===\n" + "\n".join(tool_outputs)
        sensor_input = sensor_input_str
        
    return {
        "built_tools": built_tools,
        "sensor_input": sensor_input,
        "code_sandbox_results": code_sandbox_results,
        "open_questions": open_questions
    }
