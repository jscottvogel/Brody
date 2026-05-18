import cv2
import base64
import numpy as np
import threading
from collections import deque
from langchain_core.messages import HumanMessage
from config import analytical_llm

class VisionSensor:
    def __init__(self):
        self.frame_buffer = deque(maxlen=10)
        self._background_thread = None
        self._stop_event = threading.Event()

    def capture_frame(self) -> np.ndarray:
        """
        Captures a single frame from the default webcam.
        Returns the numpy array frame, or None if failed.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Warning: Could not open the webcam.")
            return None
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Warning: Failed to capture a frame from the webcam.")
            return None
            
        return frame

    def frame_to_base64(self, frame: np.ndarray) -> str:
        """
        Converts numpy frame to base64 JPEG for Claude vision API.
        """
        if frame is None:
            return ""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def describe_scene(self, frame: np.ndarray) -> str:
        """
        Sends frame to claude-sonnet-4-20250514 with vision.
        """
        if frame is None:
            return ""
            
        b64_image = self.frame_to_base64(frame)
        prompt = 'Describe what you observe in detail. Note objects, people, spatial relationships, lighting, and anomalies.'
        
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
            }
        ]
        
        message = HumanMessage(content=content)
        try:
            response = analytical_llm.invoke([message])
            return response.content.strip()
        except Exception as e:
            print(f"Vision API error (describe_scene): {e}")
            return ""

    def detect_changes(self, frame_a: np.ndarray, frame_b: np.ndarray) -> str:
        """
        Sends both frames and asks Claude what changed between them.
        """
        if frame_a is None or frame_b is None:
            return "Missing frame(s) for comparison."
            
        b64_a = self.frame_to_base64(frame_a)
        b64_b = self.frame_to_base64(frame_b)
        
        prompt = 'Compare these two images sequentially (Before and After). What specifically has changed between them?'
        
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_a}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_b}"}}
        ]
        
        message = HumanMessage(content=content)
        try:
            response = analytical_llm.invoke([message])
            return response.content.strip()
        except Exception as e:
            print(f"Vision API error (detect_changes): {e}")
            return ""

    def start_background_capture(self, interval_seconds: int = 5):
        """
        Captures frames at interval, stores last 10 in frame_buffer.
        """
        self._stop_event.clear()
        
        def capture_loop():
            # Keep video capture open in the background thread for speed
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Warning: Background vision thread could not open webcam.")
                return
                
            print(f"\n[Vision] Background capture started (Interval: {interval_seconds}s)")
            
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if ret:
                    self.frame_buffer.append(frame)
                
                # Wait for the interval duration
                self._stop_event.wait(interval_seconds)
                
            cap.release()
            print("[Vision] Background capture stopped.")

        self._background_thread = threading.Thread(target=capture_loop, daemon=True)
        self._background_thread.start()

    def stop_background_capture(self):
        """Stops the background capture loop."""
        self._stop_event.set()
        if self._background_thread:
            self._background_thread.join()
