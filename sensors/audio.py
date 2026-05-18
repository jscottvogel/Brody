import sounddevice as sd
import wave
import tempfile
import os
import whisper
import threading
import numpy as np

class AudioSensor:
    def __init__(self, silence_threshold: int = 500):
        self.silence_threshold = silence_threshold
        self.chunk = 1024
        self.channels = 1
        self.rate = 16000 # Whisper operates on 16kHz audio
        self.is_muted = False # Used to prevent robot from transcribing its own speech
        self.transcription_buffer = []
        self._background_thread = None
        
        print("Loading Whisper model ('base'). This might take a moment if it's the first run...")
        try:
            self.model = whisper.load_model("base")
        except Exception as e:
            print(f"Warning: Failed to load whisper model: {e}")
            self.model = None

    def get_buffered_text(self) -> str:
        """Retrieves and clears the transcription buffer."""
        if not self.transcription_buffer:
            return ""
        text = " ".join(self.transcription_buffer)
        self.transcription_buffer.clear()
        return text

    def is_silence(self, audio_data: np.ndarray) -> bool:
        """Returns true if audio energy is below threshold."""
        if len(audio_data) == 0:
            return True
        rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
        return rms < self.silence_threshold

    def _transcribe_frames(self, frames: list) -> str:
        if not self.model or not frames:
            return ""
            
        audio_data = np.concatenate(frames)
        temp_path = ""
        try:
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd) 
            
            wf = wave.open(temp_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(self.rate)
            wf.writeframes(audio_data.tobytes())
            wf.close()
            
            # Transcribe using Whisper
            result = self.model.transcribe(temp_path)
            transcription = result["text"].strip()
            
        except Exception as e:
            print(f"Warning: Whisper transcription error: {e}")
            transcription = ""
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            
        return transcription

    def listen(self, duration_seconds: int = 5) -> str:
        """
        Records from default microphone for duration_seconds, 
        runs through whisper (base model) locally, 
        returns transcribed text or empty string.
        """
        if self.model is None or self.is_muted:
            return ""
            
        print(f"\n[Microphone] Listening for {duration_seconds} seconds...")
        
        try:
            recording = sd.rec(int(duration_seconds * self.rate), samplerate=self.rate, channels=self.channels, dtype='int16')
            sd.wait() # Wait until recording is finished
            print("[Microphone] Done listening.")
            
            if self.is_muted:
                return ""
                
            return self._transcribe_frames([recording])
            
        except Exception as e:
            print(f"Warning: Microphone error: {e}")
            return ""

    def start_background_listen(self):
        """
        Continuously listens, fires callback when speech detected.
        Uses silence threshold to detect end of utterance.
        """
        if self.model is None:
            print("Cannot start background listen: Whisper model failed to load.")
            return

        def listen_loop():
            print("\n[Microphone] Background listening started...")
            silence_limit_seconds = 1.5
            max_silence_chunks = int(self.rate / self.chunk * silence_limit_seconds)
            
            try:
                stream = sd.InputStream(samplerate=self.rate, channels=self.channels, dtype='int16', blocksize=self.chunk)
                with stream:
                    while True:
                        data, overflowed = stream.read(self.chunk)
                        if not self.is_muted and not self.is_silence(data):
                            # Speech started
                            frames_list = [data]
                            silence_chunks = 0
                            
                            # Record until silence
                            while True:
                                data, overflowed = stream.read(self.chunk)
                                if not self.is_muted:
                                    frames_list.append(data)
                                    if self.is_silence(data):
                                        silence_chunks += 1
                                    else:
                                        silence_chunks = 0
                                else:
                                    # Interrupted by mute (e.g. robot started speaking)
                                    frames_list = []
                                    break
                                    
                                if silence_chunks >= max_silence_chunks:
                                    break
                                    
                            # Speech ended, transcribe if frames exist
                            if frames_list:
                                text = self._transcribe_frames(frames_list)
                                if text:
                                    print(f"[Microphone] Buffered: {text}")
                                    self.transcription_buffer.append(text)
                                    
            except Exception as e:
                print(f"Warning: Background listen error: {e}")

        # Run in a background thread
        self._background_thread = threading.Thread(target=listen_loop, daemon=True)
        self._background_thread.start()
