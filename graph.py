import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import ConsciousnessState
from config import CYCLE_LIMIT

# Import nodes
from nodes.recall import recall_node
from nodes.observe import observe_node
from nodes.interpret import interpret_node
from nodes.introspect import introspect_node
from nodes.contradict import contradiction_node
from nodes.curiosity import curiosity_node
from nodes.narrative import narrative_node
from nodes.memorize import memorize_node

# Ensure checkpoints directory exists
os.makedirs("./.checkpoints", exist_ok=True)

# Build StateGraph
builder = StateGraph(ConsciousnessState)

# Add nodes
builder.add_node("recall", recall_node)
builder.add_node("observe", observe_node)
builder.add_node("interpret", interpret_node)
builder.add_node("introspect", introspect_node)
builder.add_node("contradict", contradiction_node)
builder.add_node("curiosity", curiosity_node)
builder.add_node("narrative", narrative_node)
builder.add_node("memorize", memorize_node)

# Add edges to establish the linear flow
builder.add_edge(START, "recall")
builder.add_edge("recall", "observe")
builder.add_edge("observe", "interpret")
builder.add_edge("interpret", "introspect")
builder.add_edge("introspect", "contradict")
builder.add_edge("contradict", "curiosity")
builder.add_edge("curiosity", "narrative")
builder.add_edge("narrative", "memorize")

def should_continue(state: ConsciousnessState) -> str:
    cycle_count = state.get("cycle_count", 0)
    # Check for a self_aware flag in the state (could be dynamically set by a node)
    self_aware = state.get("self_aware", False)
    
    if cycle_count >= CYCLE_LIMIT or self_aware:
        return END
    return "recall"

builder.add_conditional_edges("memorize", should_continue)

# Setup memory checkpointing
conn = sqlite3.connect("./.checkpoints/robot.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Compile and export the graph
consciousness_graph = builder.compile(checkpointer=memory)

# Config to use when invoking the graph
run_config = {"configurable": {"thread_id": "consciousness-1"}}
