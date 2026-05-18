import argparse
import sys
import os
import shutil
import json

from graph import builder, memory, run_config
from state import default_state
from config import reasoning_llm
from sensors import hub

def clear_data():
    paths_to_clear = [
        "./.checkpoints",
        "./chroma_db",
        "./chromadb",
        "./memory/chroma_db",
        "./memory/chromadb",
        "./chroma"
    ]
    cleared = False
    for path in paths_to_clear:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                cleared = True
            except Exception as e:
                print(f"Warning: Could not clear {path}: {e}")
    if cleared:
        print("Reset completed: Checkpoints and ChromaDB cleared.\n")
    else:
        print("Reset completed: No existing data found to clear.\n")

def main():
    parser = argparse.ArgumentParser(description="Run the Brody Consciousness Loop")
    parser.add_argument("--reset", action="store_true", help="Clear ChromaDB and checkpoints before running")
    args = parser.parse_args()

    if args.reset:
        clear_data()

    print("Starting consciousness loop...\n")
    
    # Startup spoken message
    hub.respond("I am becoming aware. I will now begin observing.")
    
    # Recompile with an interrupt before 'recall' so we can inject sensor input for each cycle
    consciousness_graph = builder.compile(checkpointer=memory, interrupt_before=["recall"])
    
    # Initialize state if we are starting fresh
    state_snapshot = consciousness_graph.get_state(run_config)
    if not state_snapshot.values:
        print("Initializing new consciousness state...")
        # Start graph with default state; it will pause immediately before 'recall'
        list(consciousness_graph.stream(default_state(), run_config))
        
    try:
        while True:
            state_snapshot = consciousness_graph.get_state(run_config)
            if not state_snapshot.next:
                # If there is no next node, the graph has finished (e.g. CYCLE_LIMIT reached)
                break
                
            current_state = state_snapshot.values
            cycle_count = current_state.get("cycle_count", 0)
            chat_history = current_state.get("chat_history", [])
            last_chat_count = len(chat_history)
            
            # Fetch live sensor input and format for state
            raw_input = hub.get_sensor_input()
            sensor_input = hub.format_for_state(raw_input)
                    
            # Inject new sensor input into the paused state
            curiosity_queue = current_state.get("curiosity_queue", [])
            
            # Synchronize with state.json to pick up changes made by the dashboard API
            try:
                if os.path.exists("state.json"):
                    with open("state.json", "r") as f:
                        api_state = json.load(f)
                        if "feature_requests" in api_state:
                            current_state["feature_requests"] = api_state["feature_requests"]
                        if "capability_map" in api_state:
                            current_state["capability_map"] = api_state["capability_map"]
                        if "implemented_features" in api_state:
                            current_state["implemented_features"] = api_state["implemented_features"]
                        if "chat_history" in api_state:
                            current_state["chat_history"] = api_state["chat_history"]
                            chat_history = api_state["chat_history"]
            except Exception as e:
                print(f"Warning: Failed to read state.json: {e}")

            # Inject new chat messages into the sensor string BEFORE updating state
            new_chat_messages = []
            if len(chat_history) > last_chat_count:
                for msg in chat_history[last_chat_count:]:
                    if msg.get("role") == "user":
                        new_chat_messages.append(f"The user typed a message: '{msg.get('text')}'")
            if new_chat_messages:
                sensor_input += "\n" + "\n".join(new_chat_messages)

            consciousness_graph.update_state(
                run_config, 
                {
                    "sensor_input": sensor_input,
                    "curiosity_queue": curiosity_queue,
                    "feature_requests": current_state.get("feature_requests", []),
                    "capability_map": current_state.get("capability_map", {}),
                    "implemented_features": current_state.get("implemented_features", []),
                    "chat_history": chat_history
                }
            )
            
            print(f"\n--- Cycle {cycle_count} ---")
            print(f"Sensor Input:\n{sensor_input}")
            
            # Track previous state counts to detect changes
            prev_open_questions = set(current_state.get("open_questions", []))
            prev_contradictions = len(current_state.get("contradictions", []))
            
            # Stream execution of the cycle
            for event in consciousness_graph.stream(None, run_config, stream_mode="updates"):
                for node_name, node_state in event.items():
                    print(f">> Executing: {node_name}")
                    
                    if not isinstance(node_state, dict):
                        continue
                        
                    # Print any new open questions discovered
                    if "open_questions" in node_state:
                        new_questions = set(node_state["open_questions"]) - prev_open_questions
                        if new_questions:
                            print(f"   New Open Questions added:")
                            for q in new_questions:
                                print(f"    - {q}")
                            prev_open_questions.update(node_state["open_questions"])
                            
                    # Print any new contradictions found
                    if "contradictions" in node_state:
                        contradictions = node_state["contradictions"]
                        if len(contradictions) > prev_contradictions:
                            new_contradictions = contradictions[prev_contradictions:]
                            print(f"   Contradictions Found:")
                            for c in new_contradictions:
                                print(f"    - {c.get('belief_1')} vs {c.get('belief_2')}")
                            prev_contradictions = len(contradictions)
                            
                    # Print narrative every 10 cycles and summarize it out loud
                    if node_name == "narrative" and "narrative" in node_state:
                        narrative_text = node_state["narrative"]
                        if narrative_text:
                            print(f"\n=== Current Narrative ===\n{narrative_text}\n=========================\n")
                            # Generate short summary of current robot thinking to speak aloud
                            try:
                                summary_prompt = f"Summarize this internal narrative into a single brief, spoken sentence representing your current thoughts: {narrative_text}"
                                response = reasoning_llm.invoke(summary_prompt)
                                spoken_summary = response.content.strip()
                                hub.respond(spoken_summary)
                            except Exception as e:
                                print(f"Warning: Failed to generate spoken summary: {e}")
                                
            # Drain recent speech into chat history
            # Write current snapshot to state.json for the dashboard API to read
            state_snapshot = consciousness_graph.get_state(run_config)
            final_vals = state_snapshot.values
            
            # Re-read state.json right before writing to avoid wiping out API changes
            try:
                if os.path.exists("state.json"):
                    with open("state.json", "r") as f:
                        api_state = json.load(f)
                        if "chat_history" in api_state:
                            final_vals["chat_history"] = api_state["chat_history"]
                        if "feature_requests" in api_state:
                            final_vals["feature_requests"] = api_state["feature_requests"]
                        if "capability_map" in api_state:
                            final_vals["capability_map"] = api_state["capability_map"]
                        if "implemented_features" in api_state:
                            final_vals["implemented_features"] = api_state["implemented_features"]
            except Exception as e:
                pass

            if "chat_history" not in final_vals:
                final_vals["chat_history"] = chat_history

            if hub.recent_speech:
                for speech in hub.recent_speech:
                    final_vals["chat_history"].append({
                        "role": "brody",
                        "text": speech,
                        "timestamp": "now"
                    })
                hub.recent_speech.clear()
                
            try:
                with open("state.json", "w") as f:
                    json.dump(final_vals, f, indent=2)
            except Exception as e:
                print(f"Warning: Failed to write state.json: {e}")
                            
    except KeyboardInterrupt:
        print("\n\n[KeyboardInterrupt] Terminating consciousness loop...")
        state_snapshot = consciousness_graph.get_state(run_config)
        final_state = state_snapshot.values
        final_narrative = final_state.get("narrative", "No narrative developed yet.")
        print(f"\n=== Final Narrative ===\n{final_narrative}\n=======================\n")
        sys.exit(0)

    print("\nConsciousness loop ended naturally.")
    # Print final narrative
    state_snapshot = consciousness_graph.get_state(run_config)
    final_narrative = state_snapshot.values.get("narrative", "No narrative developed yet.")
    print(f"\n=== Final Narrative ===\n{final_narrative}\n=======================\n")

if __name__ == "__main__":
    main()
