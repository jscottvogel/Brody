import pyttsx3

class Speaker:
    def __init__(self, audio_sensor=None):
        """
        Initializes the offline pyttsx3 Text-to-Speech engine.
        Accepts an optional audio_sensor to mute it while speaking.
        """
        self.engine = pyttsx3.init()
        self.audio_sensor = audio_sensor
        self._is_speaking = False
        self.set_voice_properties()

    def set_voice_properties(self, rate: int = 150, volume: float = 0.9):
        """Adjusts the speaking rate and volume of the TTS engine."""
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

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
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Warning: Failed to speak. TTS Error: {e}")
        finally:
            self._is_speaking = False
            if self.audio_sensor:
                self.audio_sensor.is_muted = False
