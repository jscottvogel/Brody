from state import ConsciousnessState

def memorize_node(state: ConsciousnessState) -> dict:
    cycle_count = state.get("cycle_count", 0)
    return {"cycle_count": cycle_count + 1}
