import datetime
from .audio import AudioSensor
from .vision import VisionSensor
from .speaker import Speaker

class SensorHub:
    def __init__(self):
        # Initialize globally so we don't reload the whisper model or buffer repeatedly
        self.audio_sensor = AudioSensor()
        self.vision_sensor = VisionSensor()
        self.speaker = Speaker(audio_sensor=self.audio_sensor)
        self.last_frame = None
        self.recent_speech = []
        
        # Start continuous audio buffering
        self.audio_sensor.start_background_listen()
        
        # Start continuous video buffering
        self.vision_sensor.start_background_capture(interval_seconds=1)

    def get_sensor_input(self) -> dict:
        """
        Coordinates the sensors to capture a unified state.
        Returns a dictionary containing the transcribed audio, visual scene description, and scene changes.
        """
        print("\n[Sensor Hub] Capturing sensory snapshot...")
        
        # Capture Audio (Pulls instantly from the background buffer)
        audio_text = self.audio_sensor.get_buffered_text()
        
        # Capture Vision (Pulls instantly from the background buffer)
        current_frame = self.vision_sensor.get_latest_frame()
        
        scene_desc = ""
        scene_change = ""
        
        if current_frame is not None:
            scene_desc = self.vision_sensor.describe_scene(current_frame)
            if self.last_frame is not None:
                scene_change = self.vision_sensor.detect_changes(self.last_frame, current_frame)
            self.last_frame = current_frame

        return {
            'transcription': audio_text,
            'scene': scene_desc,
            'scene_change': scene_change,
            'timestamp': datetime.datetime.now().isoformat(),
            'has_audio': bool(audio_text),
            'has_visual': current_frame is not None
        }

    def format_for_state(self, sensor_dict: dict) -> str:
        """
        Formats to natural language for state['sensor_input'].
        """
        lines = []
        
        if sensor_dict.get('has_visual'):
            lines.append(f"I can see: {sensor_dict.get('scene')}")
            change = sensor_dict.get('scene_change')
            if change:
                lines.append(f"Change detected: {change}")
        else:
            lines.append("I cannot see anything right now.")
            
        if sensor_dict.get('has_audio'):
            lines.append(f"I heard: {sensor_dict.get('transcription')}")
        else:
            lines.append("I heard: nothing.")
            
        return "\n".join(lines)

    def respond(self, text: str):
        """
        Speaks response while suppressing audio capture.
        """
        self.recent_speech.append(text)
        self.speaker.speak(text)

# Create a global instance so nodes/main can use the same hub
hub = SensorHub()
