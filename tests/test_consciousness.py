import pytest
from unittest.mock import MagicMock
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

# Patch the LLMs in config before importing the graph or nodes
import config

# Mock structured output
def mock_with_structured_output(schema):
    runnable = MagicMock()
    if schema.__name__ == 'IntrospectionOutput':
        from nodes.introspect import IntrospectionOutput
        runnable.invoke.return_value = IntrospectionOutput(
            updated_self_model={"identity": "robot", "status": "learning"},
            new_values=["truth"],
            new_open_questions=["What is my purpose?"]
        )
    elif schema.__name__ == 'ContradictionOutput':
        from nodes.contradict import ContradictionOutput
        runnable.invoke.return_value = ContradictionOutput(analyzed_contradictions=[])
    elif schema.__name__ == 'CuriosityOutput':
        from nodes.curiosity import CuriosityOutput
        runnable.invoke.return_value = CuriosityOutput(
            selected_question="What is my purpose?",
            exploration_targets=["Explore mirror"],
            narrative_note="I am curious about the mirror."
        )
    else:
        runnable.invoke.return_value = MagicMock()
    return runnable

# Subclass to bypass Pydantic attribute limitations
class CustomFakeChatModel(FakeMessagesListChatModel):
    def with_structured_output(self, schema, **kwargs):
        return mock_with_structured_output(schema)

# Use LangChain's fake LLM utilities for standard invokes
fake_reasoning = CustomFakeChatModel(responses=[AIMessage(content="I am reflecting on my existence.")])
fake_analytical = CustomFakeChatModel(
    responses=[AIMessage(content='{"entities": {"Mirror": "reflects"}, "emotion_tag": "curious"}')]
)

config.reasoning_llm = fake_reasoning
config.analytical_llm = fake_analytical

# Now import the graph and state
from graph import builder
from state import default_state
from langgraph.checkpoint.memory import MemorySaver

def test_consciousness_loop():
    # Use MemorySaver instead of SqliteSaver for tests
    memory = MemorySaver()
    consciousness_graph = builder.compile(checkpointer=memory, interrupt_before=["recall"])
    run_config = {"configurable": {"thread_id": "test-consciousness-1"}}
    
    # 1. Initialize a fresh state
    list(consciousness_graph.stream(default_state(), run_config))
    
    # 2. Feed 3 sequential sensor inputs
    inputs = [
        'I see my own reflection in a mirror',
        'The reflection moves when I move',
        'I am told that I am a robot'
    ]
    
    # 3. Run the full graph for those 3 cycles
    for sensor_input in inputs:
        consciousness_graph.update_state(run_config, {"sensor_input": sensor_input})
        list(consciousness_graph.stream(None, run_config))
        
    final_state = consciousness_graph.get_state(run_config).values
    
    # 4. Assertions
    assert final_state.get("world_model"), "world_model should not be empty"
    assert len(final_state.get("self_model", {})) >= 1, "self_model should contain at least one entry"
    assert final_state.get("open_questions"), "open_questions should not be empty"
    assert final_state.get("narrative"), "narrative should not be empty"
    assert final_state.get("cycle_count") == 3, f"Expected cycle_count == 3, but got {final_state.get('cycle_count')}"
