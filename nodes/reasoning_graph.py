import json
import uuid
import networkx as nx
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
import chromadb

from langchain_core.messages import SystemMessage, HumanMessage
from config import analytical_llm
from state import ConsciousnessState

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="./chromadb")
graph_collection = chroma_client.get_or_create_collection(name="reasoning_nodes")

class GraphNode(BaseModel):
    id: str = Field(description="Unique identifier for the node, e.g. 'q_1', 'goal_2'")
    type: Literal["goal", "constraint", "reasoning_state", "question"]
    content: str = Field(description="The actual content or description")
    
class GraphEdge(BaseModel):
    from_id: str
    to_id: str
    relationship: Literal["depends_on", "blocks", "enables", "questions", "contradicts", "requires"]

class GraphExtractionOutput(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)

def reasoning_graph_node(state: ConsciousnessState) -> dict:
    cycle_count = state.get("cycle_count", 0)
    open_questions = state.get("open_questions", [])
    beliefs = state.get("beliefs", [])
    contradictions = state.get("contradictions", [])
    goal_states = state.get("goal_states", [])
    reasoning_history = state.get("reasoning_history", [])
    
    recent_history = reasoning_history[-3:] if reasoning_history else []
    
    system_prompt = (
        "You are the meta-cognitive graph mapping module. Your job is to extract new cognitive nodes "
        "(goals, constraints, reasoning_states, questions) and edges (relationships) from the current state."
    )
    
    human_prompt = (
        f"Cycle: {cycle_count}\n"
        f"Open Questions: {json.dumps(open_questions, indent=2)}\n"
        f"Beliefs: {json.dumps(beliefs, indent=2)}\n"
        f"Contradictions: {json.dumps(contradictions, indent=2)}\n"
        f"Goal States: {json.dumps(goal_states, indent=2)}\n"
        f"Recent Reasoning History: {json.dumps(recent_history, indent=2)}\n\n"
        "Identify any NEW distinct graph nodes and how they connect to each other. "
        "Use simple, consistent node IDs (e.g. 'q1', 'constraint_a')."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    structured_llm = analytical_llm.with_structured_output(GraphExtractionOutput)
    
    try:
        response = structured_llm.invoke(messages)
        if not response:
            return {}
    except Exception as e:
        print(f"Warning: reasoning_graph LLM error: {e}")
        return {}
        
    # Get current graph
    current_graph = state.get("reasoning_graph", {"nodes": [], "edges": []})
    nodes_list = list(current_graph.get("nodes", []))
    edges_list = list(current_graph.get("edges", []))
    
    existing_node_ids = {n["id"] for n in nodes_list}
    
    new_docs = []
    new_ids = []
    new_metadatas = []
    
    for n in response.nodes:
        if n.id not in existing_node_ids:
            node_dict = {
                "id": n.id,
                "type": n.type,
                "content": n.content,
                "cycle_created": cycle_count,
                "resolved": False
            }
            nodes_list.append(node_dict)
            existing_node_ids.add(n.id)
            
            # Prepare for ChromaDB
            new_ids.append(f"cycle_{cycle_count}_{n.id}")
            new_docs.append(n.content)
            new_metadatas.append({"type": n.type, "node_id": n.id, "cycle_created": cycle_count})
            
    # Add new docs to chroma
    if new_docs:
        try:
            graph_collection.add(
                documents=new_docs,
                metadatas=new_metadatas,
                ids=new_ids
            )
        except Exception as e:
            print(f"Warning: Failed to save nodes to ChromaDB: {e}")
            
    for e in response.edges:
        duplicate = any(ex["from_id"] == e.from_id and ex["to_id"] == e.to_id and ex["relationship"] == e.relationship for ex in edges_list)
        if not duplicate:
            edges_list.append({
                "from_id": e.from_id,
                "to_id": e.to_id,
                "relationship": e.relationship
            })
            
    updated_graph = {"nodes": nodes_list, "edges": edges_list}
    
    # Check for cycles using NetworkX
    G = nx.DiGraph()
    for n in nodes_list:
        G.add_node(n["id"])
    for e in edges_list:
        G.add_edge(e["from_id"], e["to_id"], relationship=e["relationship"])
        
    detected_loops = list(state.get("detected_loops", []))
    existing_loop_paths = [tuple(l["nodes_involved"]) for l in detected_loops]
    
    try:
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            # We only want cycles that haven't been recorded
            # Simple check, though cycle permutations can exist
            cycle_tuple = tuple(cycle)
            if cycle_tuple not in existing_loop_paths:
                new_loop = {
                    "id": str(uuid.uuid4()),
                    "loop_type": "circular_dependency",
                    "nodes_involved": cycle,
                    "first_detected_cycle": cycle_count,
                    "times_repeated": 0,
                    "escape_conditions": [],
                    "escape_status": "identified"
                }
                detected_loops.append(new_loop)
                existing_loop_paths.append(cycle_tuple)
    except Exception as e:
        print(f"Warning: Cycle detection failed: {e}")
        
    meta_cap = dict(state.get("meta_cognitive_capacity", {}))
    meta_cap["can_detect_loops"] = True
    
    new_reasoning_entry = {
        "cycle": cycle_count,
        "node_id": "reasoning_graph",
        "reasoning_summary": f"Extracted {len(response.nodes)} new nodes and {len(response.edges)} new edges.",
        "question_asked": open_questions[-1] if open_questions else "None",
        "conclusion_reached": beliefs[-1].get("content") if beliefs and isinstance(beliefs[-1], dict) else str(beliefs[-1]) if beliefs else None,
        "led_to": [n.id for n in response.nodes]
    }
    updated_history = list(reasoning_history)
    updated_history.append(new_reasoning_entry)
    
    return {
        "reasoning_graph": updated_graph,
        "detected_loops": detected_loops,
        "meta_cognitive_capacity": meta_cap,
        "reasoning_history": updated_history
    }
