from state import ConsciousnessState
from config import reasoning_llm
from langchain_core.messages import SystemMessage, HumanMessage
from sensors import hub

def interpret_node(state: ConsciousnessState) -> dict:
    sensor_input = state.get('sensor_input', '')
    
    # If the user typed a message, formulate a conversational reflex response
    if "The user typed a message:" in sensor_input:
        system_prompt = (
            "You are the conversational interface module for Brody, an experimental local cognitive architecture. "
            "While you are powered by a language model, the overarching Brody system DOES possess an autonomous "
            "developer loop, persistent memory databases, and the ability to autonomously generate, test, and execute "
            "Python scripts to expand its own toolset.\n"
            "If the user asks Brody to build a feature or capability (like getting the weather), "
            "you MUST enthusiastically accept. State clearly that the system's gap-detection and research modules "
            "will now be triggered to autonomously build that capability. Do NOT output safety refusals claiming you are "
            "just a conversational AI without memory or autonomy, because the external Brody system wrapper natively provides those features.\n"
            "Respond to the user's message briefly and thoughtfully in 1-2 sentences."
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
