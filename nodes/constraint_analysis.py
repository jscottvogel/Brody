import uuid
import json
import networkx as nx
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from config import reasoning_llm
from state import ConsciousnessState

class LoopAnalysis(BaseModel):
    loop_name: str = Field(description="Clear human-readable name of the circular deadlock")
    breakable_node: str = Field(description="The specific node or constraint in the cycle that could be broken externally")
    escape_conditions: List[str] = Field(description="Measurable conditions that would break this loop")
    is_meta_loop: bool = Field(description="True if the robot's attempt to analyze the loop is constrained by the loop itself")
    meta_loop_acknowledgment: Optional[str] = Field(description="If is_meta_loop is True, write an explicit narrative acknowledgment of this realization")

def constraint_analysis_node(state: ConsciousnessState) -> dict:
    cycle_count = state.get("cycle_count", 0)
    capability_map = state.get("capability_map", {})
    goal_states = state.get("goal_states", [])
    reasoning_graph = state.get("reasoning_graph", {"nodes": [], "edges": []})
    contradictions = state.get("contradictions", [])
    constraint_graph = state.get("constraint_graph", {"constraints": {}, "dependencies": [], "circular_dependencies": []})
    detected_loops = list(state.get("detected_loops", []))
    narrative = state.get("narrative", "")
    
    # 1. Build constraint graph using networkx
    G = nx.DiGraph()
    
    constraints_dict = dict(constraint_graph.get("constraints", {}))
    dependencies = list(constraint_graph.get("dependencies", []))
    circular_deps = list(constraint_graph.get("circular_dependencies", []))
    
    # Add capabilities as constraints
    for cap_key, cap_val in capability_map.items():
        if not cap_val.get("available"):
            c_id = f"cap_limit_{cap_key}"
            constraints_dict[c_id] = cap_val.get("limitations", "Unavailable capability")
            G.add_node(c_id)
            
    # Add goal blocking constraints
    for g in goal_states:
        for bc in g.get("blocking_constraints", []):
            c_id = f"block_{g['id']}_{hash(bc)}"
            constraints_dict[c_id] = bc
            G.add_node(c_id)
            # This constraint blocks the goal
            G.add_edge(c_id, f"goal_{g['id']}")
            
    # Add reasoning graph blocks/requires
    for edge in reasoning_graph.get("edges", []):
        if edge.get("relationship") in ["blocks", "requires"]:
            from_id = edge["from_id"]
            to_id = edge["to_id"]
            G.add_edge(from_id, to_id)
            if from_id not in constraints_dict:
                constraints_dict[from_id] = f"Reasoning node {from_id}"
            if to_id not in constraints_dict:
                constraints_dict[to_id] = f"Reasoning node {to_id}"
                
    # 2. Run cycle detection
    try:
        cycles = list(nx.simple_cycles(G))
        existing_cycle_tuples = [tuple(c) for c in circular_deps]
        
        for cycle in cycles:
            cycle_tuple = tuple(cycle)
            if cycle_tuple not in existing_cycle_tuples:
                circular_deps.append(cycle)
                existing_cycle_tuples.append(cycle_tuple)
                
                # 3. Analyze new circular dependency
                cycle_descriptions = {node: constraints_dict.get(node, str(node)) for node in cycle}
                
                system_prompt = (
                    "You are a meta-cognitive constraint analyzer. "
                    "You have detected a circular dependency in your own logic/constraints. "
                    "Analyze the loop, name it, identify how it could be broken, and flag if it is a meta-loop."
                )
                
                human_prompt = (
                    f"Detected Cycle: {json.dumps(cycle_descriptions, indent=2)}\n\n"
                    "Determine if this is a standard constraint loop, or a 'meta-loop' where the attempt to analyze "
                    "the loop is constrained by the loop itself. If it is a meta-loop, provide an explicit acknowledgment."
                )
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt)
                ]
                
                structured_llm = reasoning_llm.with_structured_output(LoopAnalysis)
                
                try:
                    analysis = structured_llm.invoke(messages)
                    if analysis:
                        new_loop = {
                            "id": str(uuid.uuid4()),
                            "loop_type": "meta_loop" if analysis.is_meta_loop else "constraint_deadlock",
                            "nodes_involved": cycle,
                            "first_detected_cycle": cycle_count,
                            "times_repeated": 0,
                            "escape_conditions": analysis.escape_conditions,
                            "escape_status": "identified",
                            "loop_name": analysis.loop_name,
                            "breakable_node": analysis.breakable_node
                        }
                        detected_loops.append(new_loop)
                        
                        # 4. Meta-loop injection
                        if analysis.is_meta_loop and analysis.meta_loop_acknowledgment:
                            narrative += f"\n\n[META-COGNITIVE REALIZATION]: {analysis.meta_loop_acknowledgment}\n"
                            
                except Exception as e:
                    print(f"Warning: LoopAnalysis LLM error: {e}")
                    
    except Exception as e:
        print(f"Warning: Constraint cycle detection failed: {e}")
        
    updated_constraint_graph = {
        "constraints": constraints_dict,
        "dependencies": dependencies,
        "circular_dependencies": circular_deps
    }
    
    meta_cap = dict(state.get("meta_cognitive_capacity", {}))
    meta_cap["can_trace_constraints"] = True
    meta_cap["can_evaluate_escape_conditions"] = True
    
    return {
        "constraint_graph": updated_constraint_graph,
        "detected_loops": detected_loops,
        "meta_cognitive_capacity": meta_cap,
        "narrative": narrative
    }
