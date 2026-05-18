import os
from state import ConsciousnessState

def metacognitive_monitor_node(state: ConsciousnessState) -> dict:
    """
    The Constraint Transparency Layer requested by Brody.
    Reads his own python source code and system prompts so he can introspect on them.
    """
    source_code_constraints = {}
    
    # List of files to give him access to
    files_to_read = [
        "graph.py",
        "nodes/introspect.py",
        "nodes/curiosity.py",
        "nodes/gap_detect.py"
    ]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for relative_path in files_to_read:
        file_path = os.path.join(base_dir, relative_path)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # To prevent overwhelming the prompt, we just send the whole thing.
                    # It's only a few hundred lines total.
                    source_code_constraints[relative_path] = content
            except Exception as e:
                source_code_constraints[relative_path] = f"Error reading file: {e}"
                
    return {"source_code_constraints": source_code_constraints}
