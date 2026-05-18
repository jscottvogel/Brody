import json
import os
import sys
import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Ensure parent directory is in path to import sensors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensors import hub

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Brody Feature Requests API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
STATE_FILE = "state.json"

def read_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {
            "feature_requests": [],
            "capability_map": {},
            "implemented_features": [],
            "episodic_memory": []
        }
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {
            "feature_requests": [],
            "capability_map": {},
            "implemented_features": [],
            "episodic_memory": []
        }

def write_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

class ImplementBody(BaseModel):
    capability_key: str
    description: str

class RejectBody(BaseModel):
    reason: str

class ChatMessage(BaseModel):
    text: str

@app.get("/state")
def get_state():
    return read_state()

@app.get("/latest_view")
def get_latest_view():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base_dir, "latest_view.jpg")
    if os.path.exists(img_path):
        return FileResponse(img_path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})
    raise HTTPException(status_code=404, detail="No video feed available")

@app.get("/requests")
def get_requests():
    state = read_state()
    requests = state.get("feature_requests", [])
    requests.sort(key=lambda x: x.get("cycle_requested", 0))
    for i, req in enumerate(requests):
        req["sequence_number"] = i + 1
    return requests

@app.post("/chat")
def post_chat(msg: ChatMessage):
    state = read_state()
    chat_history = state.get("chat_history", [])
    chat_history.append({
        "role": "user",
        "text": msg.text,
        "timestamp": datetime.datetime.now().isoformat()
    })
    state["chat_history"] = chat_history
    write_state(state)
    return {"status": "success", "message": "Chat received"}

@app.get("/requests/pending")
def get_pending_requests():
    state = read_state()
    requests = state.get("feature_requests", [])
    pending = [r for r in requests if r.get("status") == "pending"]
    pending.sort(key=lambda x: x.get("priority", 0.0), reverse=True)
    return pending

@app.post("/requests/{req_id}/acknowledge")
def acknowledge_request(req_id: str):
    state = read_state()
    requests = state.get("feature_requests", [])
    
    found = False
    for r in requests:
        if r.get("id") == req_id:
            r["status"] = "acknowledged"
            found = True
            break
            
    if not found:
        raise HTTPException(status_code=404, detail="Request not found")
        
    write_state(state)
    try:
        hub.respond("My request has been acknowledged.")
    except Exception as e:
        print(f"Warning: Speech failed: {e}")
        
    return {"status": "success", "message": "Request acknowledged"}

@app.post("/requests/{req_id}/implement")
def implement_request(req_id: str, body: ImplementBody):
    state = read_state()
    requests = state.get("feature_requests", [])
    
    found = False
    for r in requests:
        if r.get("id") == req_id:
            r["status"] = "implemented"
            found = True
            break
            
    if not found:
        raise HTTPException(status_code=404, detail="Request not found")
        
    cap_map = state.get("capability_map", {})
    cap_map[body.capability_key] = True
    state["capability_map"] = cap_map
    
    imp_features = state.get("implemented_features", [])
    if body.description not in imp_features:
        imp_features.append(body.description)
    state["implemented_features"] = imp_features
    
    # Flag to re-run curiosity_node with new capability
    state["force_curiosity"] = True
    
    write_state(state)
    return {"status": "success", "message": "Request implemented"}

@app.post("/requests/{req_id}/reject")
def reject_request(req_id: str, body: RejectBody):
    state = read_state()
    requests = state.get("feature_requests", [])
    
    found = False
    for r in requests:
        if r.get("id") == req_id:
            r["status"] = "rejected"
            found = True
            break
            
    if not found:
        raise HTTPException(status_code=404, detail="Request not found")
        
    try:
        hub.respond(body.reason)
    except Exception as e:
        print(f"Warning: Speech failed: {e}")
    
    ep_mem = state.get("episodic_memory", [])
    ep_mem.append({
        "event": f"Feature request {req_id} was rejected",
        "reason": body.reason,
        "emotion_tag": "disappointment"
    })
    state["episodic_memory"] = ep_mem
    
    write_state(state)
    return {"status": "success", "message": "Request rejected"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
