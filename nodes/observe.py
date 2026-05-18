import json
from langchain_core.messages import SystemMessage, HumanMessage
from config import analytical_llm
from state import ConsciousnessState

def observe_node(state: ConsciousnessState) -> dict:
    sensor_input = state.get('sensor_input', '')
    
    if not sensor_input:
        return {}
        
    text_input = str(sensor_input)
    
    reasoning_graph = state.get("reasoning_graph", {})
    detected_loops = state.get("detected_loops", [])
    loop_context = f"\n\nCURRENT LOOPS AND COGNITIVE CONSTRAINTS:\n- Detected Loops: {json.dumps(detected_loops, indent=2)}\n- Cognitive Graph Nodes: {len(reasoning_graph.get('nodes', []))}" if detected_loops else ""

    prompt = f"""
    You are an analytical observation module for a cognitive architecture.
    Analyze the following raw sensory input: "{text_input}"
    
    Extract structured facts and respond ONLY with valid JSON matching this schema exactly:
    {{
        "entities": {{
            "EntityName": "Description of properties"
        }},
        "events": [
            "Description of event 1"
        ],
        "anomalies": [
            "Description of any unexpected or anomalous observation"
        ],
        "emotion_tag": "A single word representing the dominant emotion (e.g., curious, alarmed, calm, confused)"
    }}
    
    Do not include markdown code blocks or any other text, just the raw JSON string.
    {loop_context}
    """
    
    messages = [
        SystemMessage(content="You are a strict data extraction system. Output strictly valid JSON without any markdown formatting."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = analytical_llm.invoke(messages)
    except Exception as e:
        print(f"Warning: observe LLM error: {e}")
        return {}
    
    try:
        content = response.content.strip()
        # Clean up any potential markdown formatting the LLM might stubbornly include
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
            
        parsed = json.loads(content.strip())
    except Exception as e:
        print(f"Failed to parse JSON from observe_node: {e}")
        parsed = {
            "entities": {},
            "events": [],
            "anomalies": [],
            "emotion_tag": "confused"
        }
        
    # 1. Update world_model
    # Since we defined ConsciousnessState as a standard TypedDict, we merge here.
    new_world_model = dict(state.get('world_model', {}))
    
    for entity, props in parsed.get("entities", {}).items():
        new_world_model[entity] = props
        
    # We can store events as temporary world_model facts or as a list under an 'events' key
    if parsed.get("events"):
        new_world_model["current_events"] = parsed.get("events")

    # 2. Episodic memory entry
    new_memory = {
        "event": text_input,
        "emotion_tag": parsed.get("emotion_tag", "neutral")
    }
    episodic_memory = list(state.get('episodic_memory', []))
    episodic_memory.append(new_memory)
    
    # 3. Anomalies to open_questions
    open_questions = list(state.get('open_questions', []))
    for anomaly in parsed.get("anomalies", []):
        open_questions.append(f"Why is this anomaly occurring: {anomaly}?")
        
    return {
        "world_model": new_world_model,
        "episodic_memory": episodic_memory,
        "open_questions": open_questions
    }
