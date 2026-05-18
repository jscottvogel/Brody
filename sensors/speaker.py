import pyttsx3
import subprocess
import sys

class Speaker:
    def __init__(self, audio_sensor=None):
        """
        Initializes the offline pyttsx3 Text-to-Speech engine.
        Accepts an optional audio_sensor to mute it while speaking.
        """
        self.audio_sensor = audio_sensor
        self._is_speaking = False

    def is_speaking(self) -> bool:
        """Returns whether speech is currently playing."""
        return self._is_speaking

    def speak(self, text: str):
        """
        Uses offline pyttsx3 to speak the given text aloud.
        Mutes the audio_sensor to prevent the robot from transcribing its own speech.
        This call blocks until the speech has finished playing.
        """
        self._is_speaking = True
        if self.audio_sensor:
            self.audio_sensor.is_muted = True
            
        try:
            print(f"[Speaker] {text}")
            # Run in an entirely separate process to completely isolate SAPI5 COM state
            script = f'''
import pyttsx3
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    engine.say({repr(text)})
    engine.runAndWait()
except Exception as e:
    pass
'''
            subprocess.run([sys.executable, "-c", script], check=False)
        except Exception as e:
            print(f"Warning: Failed to speak. TTS Error: {e}")
        finally:
            self._is_speaking = False
            if self.audio_sensor:
                self.audio_sensor.is_muted = False
