import pyaudio
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
        self.format_type = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000 # Whisper operates on 16kHz audio
        self.is_muted = False # Used to prevent robot from transcribing its own speech
        
        print("Loading Whisper model ('base'). This might take a moment if it's the first run...")
        try:
            self.model = whisper.load_model("base")
        except Exception as e:
            print(f"Warning: Failed to load whisper model: {e}")
            self.model = None

    def is_silence(self, audio_data: bytes) -> bool:
        """Returns true if audio energy is below threshold."""
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        if len(audio_np) == 0:
            return True
        rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
        return rms < self.silence_threshold

    def _transcribe_frames(self, frames: list) -> str:
        if not self.model or not frames:
            return ""
            
        p = pyaudio.PyAudio()
        temp_path = ""
        try:
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd) 
            
            wf = wave.open(temp_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.format_type))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Transcribe using Whisper
            result = self.model.transcribe(temp_path)
            transcription = result["text"].strip()
            
        except Exception as e:
            print(f"Warning: Whisper transcription error: {e}")
            transcription = ""
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            p.terminate()
            
        return transcription

    def listen(self, duration_seconds: int = 5) -> str:
        """
        Records from default microphone for duration_seconds, 
        runs through whisper (base model) locally, 
        returns transcribed text or empty string.
        """
        if self.model is None or self.is_muted:
            return ""
            
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(format=self.format_type,
                            channels=self.channels,
                            rate=self.rate,
                            input=True,
                            frames_per_buffer=self.chunk)
            
            print(f"\n[Microphone] Listening for {duration_seconds} seconds...")
            
            frames = []
            for _ in range(0, int(self.rate / self.chunk * duration_seconds)):
                data = stream.read(self.chunk)
                if not self.is_muted:
                    frames.append(data)
                
            print("[Microphone] Done listening.")
            
        except Exception as e:
            print(f"Warning: Microphone error: {e}")
            return ""
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
            p.terminate()

        if self.is_muted:
            return ""

        return self._transcribe_frames(frames)

    def start_background_listen(self, callback):
        """
        Continuously listens, fires callback when speech detected.
        Uses silence threshold to detect end of utterance.
        """
        if self.model is None:
            print("Cannot start background listen: Whisper model failed to load.")
            return

        def listen_loop():
            p = pyaudio.PyAudio()
            try:
                stream = p.open(format=self.format_type,
                                channels=self.channels,
                                rate=self.rate,
                                input=True,
                                frames_per_buffer=self.chunk)
                
                print("\n[Microphone] Background listening started...")
                
                silence_limit_seconds = 1.5
                max_silence_chunks = int(self.rate / self.chunk * silence_limit_seconds)
                
                while True:
                    # Wait for speech to start
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    if not self.is_muted and not self.is_silence(data):
                        # Speech started
                        frames = [data]
                        silence_chunks = 0
                        
                        # Record until silence
                        while True:
                            data = stream.read(self.chunk, exception_on_overflow=False)
                            if not self.is_muted:
                                frames.append(data)
                                if self.is_silence(data):
                                    silence_chunks += 1
                                else:
                                    silence_chunks = 0
                            else:
                                # Interrupted by mute (e.g. robot started speaking)
                                frames = []
                                break
                                
                            if silence_chunks >= max_silence_chunks:
                                break
                                
                        # Speech ended, transcribe if frames exist
                        if frames:
                            text = self._transcribe_frames(frames)
                            if text:
                                callback(text)
                            
            except Exception as e:
                print(f"Warning: Background listen error: {e}")
            finally:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
                p.terminate()

        # Run in a background thread
        thread = threading.Thread(target=listen_loop, daemon=True)
        thread.start()
