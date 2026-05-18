from state import ConsciousnessState
from config import reasoning_llm
from langchain_core.messages import SystemMessage, HumanMessage
from sensors import hub

def interpret_node(state: ConsciousnessState) -> dict:
    sensor_input = state.get('sensor_input', '')
    
    # If the user typed a message, formulate a conversational reflex response
    if "The user typed a message:" in sensor_input:
        system_prompt = (
            "You are Brody, a cognitive architecture. You have just observed a chat message from the user. "
            "You are not a traditional chatbot, but an autonomous entity. "
            "Respond to the user's message briefly and thoughtfully in 1-2 sentences. "
            "Speak your mind directly."
        )
        human_prompt = f"Recent Sensory Input:\n{sensor_input}\n\nPlease generate your spoken response:"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = reasoning_llm.invoke(messages)
            spoken_response = response.content.strip()
            if spoken_response:
                hub.respond(spoken_response)
        except Exception as e:
            print(f"Warning: interpret LLM error: {e}")
            
    return {}
