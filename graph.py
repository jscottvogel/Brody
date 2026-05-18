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
from nodes.metacognitive_monitor import metacognitive_monitor_node
from nodes.introspect import introspect_node
from nodes.goal_state import goal_state_node
from nodes.reasoning_graph import reasoning_graph_node
from nodes.constraint_analysis import constraint_analysis_node
from nodes.pattern_detect import pattern_detect_node
from nodes.escape_analysis import escape_analysis_node
from nodes.contradict import contradiction_node
from nodes.curiosity import curiosity_node
from nodes.gap_detect import gap_detect_node
from nodes.feature_request import feature_request_node
from nodes.evaluate_features import evaluate_features_node
from nodes.narrative import narrative_node
from nodes.memorize import memorize_node

# Import Developer Loop Nodes
from nodes.tool_use import tool_use_node
from nodes.research import research_node
from nodes.evaluate import evaluate_node
from nodes.api_discover import api_discover_node
from nodes.scaffold import scaffold_node
from nodes.sandbox import sandbox_node

# Ensure checkpoints directory exists
os.makedirs("./.checkpoints", exist_ok=True)

# Build StateGraph
builder = StateGraph(ConsciousnessState)

# Add nodes
builder.add_node("recall", recall_node)
builder.add_node("observe", observe_node)
builder.add_node("interpret", interpret_node)
builder.add_node("metacognitive_monitor", metacognitive_monitor_node)
builder.add_node("introspect", introspect_node)
builder.add_node("goal_state", goal_state_node)
builder.add_node("reasoning_graph", reasoning_graph_node)
builder.add_node("constraint_analysis", constraint_analysis_node)
builder.add_node("pattern_detect", pattern_detect_node)
builder.add_node("escape_analysis", escape_analysis_node)
builder.add_node("contradict", contradiction_node)
builder.add_node("curiosity", curiosity_node)
builder.add_node("gap_detect", gap_detect_node)
builder.add_node("feature_request", feature_request_node)
builder.add_node("evaluate_features", evaluate_features_node)
builder.add_node("narrative", narrative_node)
builder.add_node("memorize", memorize_node)

# Add Developer Loop Nodes
builder.add_node("tool_use", tool_use_node)
builder.add_node("research", research_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("api_discover", api_discover_node)
builder.add_node("scaffold", scaffold_node)
builder.add_node("sandbox", sandbox_node)

# Add edges to establish the linear flow
builder.add_edge(START, "recall")
builder.add_edge("recall", "tool_use")
builder.add_edge("tool_use", "observe")
builder.add_edge("observe", "interpret")
builder.add_edge("interpret", "metacognitive_monitor")
builder.add_edge("metacognitive_monitor", "introspect")
builder.add_edge("introspect", "reasoning_graph")
builder.add_edge("reasoning_graph", "pattern_detect")
builder.add_edge("pattern_detect", "constraint_analysis")
builder.add_edge("constraint_analysis", "goal_state")
builder.add_edge("goal_state", "escape_analysis")

def route_after_escape(state: ConsciousnessState) -> str:
    em = state.get("episodic_memory", [])
    if em and em[-1].get("emotion_tag") == "insight":
        return "narrative"
    return "contradict"

builder.add_conditional_edges("escape_analysis", route_after_escape)
builder.add_edge("contradict", "curiosity")
builder.add_edge("curiosity", "gap_detect")

# Developer Loop Conditional Routing
def route_after_gap_detect(state: ConsciousnessState) -> str:
    research_queue = state.get("research_queue", [])
    cycle_count = state.get("cycle_count", 0)
    
    if cycle_count > 0 and cycle_count % 5 == 0 and any(item.get("status") == "queued" for item in research_queue):
        return "research"
    return "feature_request"

builder.add_conditional_edges("gap_detect", route_after_gap_detect)

# Developer Loop Execution Chain
builder.add_edge("research", "evaluate")
builder.add_edge("evaluate", "api_discover")
builder.add_edge("api_discover", "scaffold")
builder.add_edge("scaffold", "sandbox")

def route_after_sandbox(state: ConsciousnessState) -> str:
    research_queue = list(state.get("research_queue", []))
    for item in research_queue:
        status = item.get("status")
        if status == "integrated":
            return "narrative"
        elif status == "scaffolding":
            return "scaffold"
        elif status == "failed":
            return "narrative"
    return "narrative"

builder.add_conditional_edges("sandbox", route_after_sandbox)

# Main Loop Continuation
builder.add_edge("feature_request", "evaluate_features")
builder.add_edge("evaluate_features", "narrative")
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
