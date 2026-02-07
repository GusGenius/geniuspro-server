"""
GeniusPro Voice Server
Real-time speech-to-speech API using WebSocket.

Pipeline: Mic → VAD → Whisper (STT) → Ollama (Brain) → VibeVoice (TTS) → Speaker
"""

import asyncio
import copy
import io
import json
import os
import struct
import threading
import traceback
import wave
from pathlib import Path
from queue import Empty, Queue
from typing import Iterator, Optional

import aiohttp
import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

# -- STT --
from faster_whisper import WhisperModel

# -- VAD --
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio

# -- TTS --
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.streamer import AudioStreamer

# ─── Config ───────────────────────────────────────────────────────────────────

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "geniuspro-coder-v1")
VIBEVOICE_MODEL = os.environ.get("VIBEVOICE_MODEL", "microsoft/VibeVoice-Realtime-0.5B")
VOICE_PRESET = os.environ.get("VOICE_PRESET", "en-Carter_man")

INPUT_SAMPLE_RATE = 16000   # Client sends 16kHz PCM16 mono
OUTPUT_SAMPLE_RATE = 24000  # VibeVoice outputs 24kHz

# VAD settings
VAD_THRESHOLD = 0.5
SILENCE_DURATION_MS = 700   # ms of silence before we consider speech ended
MIN_SPEECH_DURATION_MS = 250  # ignore very short noises

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="GeniusPro Voice")

BASE = Path(__file__).parent


# ─── TTS Service ──────────────────────────────────────────────────────────────

class TTSService:
    """Wraps VibeVoice-Realtime for streaming TTS."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.sample_rate = OUTPUT_SAMPLE_RATE
        self.processor = None
        self.model = None
        self.voice_presets = {}
        self.default_voice_key = None
        self._voice_cache = {}

    def load(self):
        print(f"[TTS] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        load_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        device_map = self.device

        # Try flash_attention_2 first, fall back to sdpa (needed for RTX 5090)
        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation="flash_attention_2",
            )
            print("[TTS] Loaded with flash_attention_2")
        except Exception:
            print("[TTS] flash_attention_2 not available, using sdpa")
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation="sdpa",
            )

        self.model.eval()
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=5)

        # Load voice presets
        voices_dir = Path(self.model_path)
        if not voices_dir.exists():
            # HuggingFace model - look in the VibeVoice repo
            voices_dir = Path.home() / "VibeVoice" / "voices" / "streaming_model"

        if voices_dir.exists():
            for pt_path in voices_dir.rglob("*.pt"):
                self.voice_presets[pt_path.stem] = pt_path
            print(f"[TTS] Found {len(self.voice_presets)} voice presets")

        self.default_voice_key = VOICE_PRESET if VOICE_PRESET in self.voice_presets else next(iter(self.voice_presets), None)

        if self.default_voice_key:
            self._load_voice(self.default_voice_key)
        print(f"[TTS] Ready, default voice: {self.default_voice_key}")

    def _load_voice(self, key: str):
        if key not in self._voice_cache and key in self.voice_presets:
            self._voice_cache[key] = torch.load(
                self.voice_presets[key],
                map_location=torch.device(self.device),
                weights_only=False,
            )

    def stream(self, text: str, voice_key: str = None, stop_event: threading.Event = None) -> Iterator[np.ndarray]:
        """Generate audio chunks from text. Yields numpy float32 arrays."""
        if not text.strip():
            return

        text = text.replace("\u2019", "'")
        key = voice_key if voice_key and voice_key in self.voice_presets else self.default_voice_key
        self._load_voice(key)
        prefilled_outputs = self._voice_cache[key]

        processor_kwargs = {
            "text": text.strip(),
            "cached_prompt": prefilled_outputs,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }
        processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)
        inputs = {
            k: v.to(torch.device(self.device)) if hasattr(v, "to") else v
            for k, v in processed.items()
        }

        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors = []
        stop_signal = stop_event or threading.Event()

        def run_generation():
            try:
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=1.5,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False, "temperature": 1.0, "top_p": 1.0},
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop_signal.is_set,
                    verbose=False,
                    refresh_negative=True,
                    all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
                )
            except Exception as exc:
                errors.append(exc)
                traceback.print_exc()
                audio_streamer.end()

        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()

        try:
            stream = audio_streamer.get_stream(0)
            for chunk in stream:
                if torch.is_tensor(chunk):
                    chunk = chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    chunk = np.asarray(chunk, dtype=np.float32)
                if chunk.ndim > 1:
                    chunk = chunk.reshape(-1)
                peak = np.max(np.abs(chunk)) if chunk.size else 0.0
                if peak > 1.0:
                    chunk = chunk / peak
                yield chunk.astype(np.float32, copy=False)
        finally:
            stop_signal.set()
            audio_streamer.end()
            thread.join()

    @staticmethod
    def chunk_to_pcm16(chunk: np.ndarray) -> bytes:
        chunk = np.clip(chunk, -1.0, 1.0)
        return (chunk * 32767.0).astype(np.int16).tobytes()


# ─── STT Service ──────────────────────────────────────────────────────────────

class STTService:
    """Wraps faster-whisper for speech-to-text."""

    def __init__(self, model_size: str = "large-v3-turbo", device: str = "cuda"):
        self.model_size = model_size
        self.device = device
        self.model = None

    def load(self):
        print(f"[STT] Loading Whisper model: {self.model_size}")
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=compute_type)
        print("[STT] Ready")

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe PCM16 mono 16kHz audio bytes to text."""
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(audio_np, beam_size=1, language="en")
        text = " ".join(seg.text for seg in segments).strip()
        return text


# ─── Ollama Client ────────────────────────────────────────────────────────────

async def ollama_chat_stream(messages: list, model: str = OLLAMA_MODEL):
    """Stream chat response from Ollama, yielding text chunks."""
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            async for line in resp.content:
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue


# ─── VAD Processor ────────────────────────────────────────────────────────────

class VADProcessor:
    """Processes incoming audio chunks and detects speech boundaries."""

    def __init__(self):
        self.model = load_silero_vad()
        self.buffer = bytearray()
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        # At 16kHz, 512 samples = 32ms per frame
        self.frame_size = 512
        self.samples_per_ms = INPUT_SAMPLE_RATE / 1000
        self.silence_frames_threshold = int(SILENCE_DURATION_MS * self.samples_per_ms / self.frame_size)
        self.min_speech_frames = int(MIN_SPEECH_DURATION_MS * self.samples_per_ms / self.frame_size)

    def reset(self):
        self.buffer = bytearray()
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0

    def process_chunk(self, pcm_bytes: bytes) -> Optional[bytes]:
        """
        Feed PCM16 mono 16kHz audio chunk.
        Returns complete speech audio bytes when speech ends, or None.
        """
        self.buffer.extend(pcm_bytes)

        # Process in 512-sample frames (32ms at 16kHz)
        bytes_per_frame = self.frame_size * 2  # 16-bit = 2 bytes per sample
        result = None

        while len(self.buffer) >= bytes_per_frame:
            frame_bytes = bytes(self.buffer[:bytes_per_frame])
            self.buffer = self.buffer[bytes_per_frame:]

            frame_np = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            frame_tensor = torch.from_numpy(frame_np)

            confidence = self.model(frame_tensor, INPUT_SAMPLE_RATE).item()

            if confidence >= VAD_THRESHOLD:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_buffer = bytearray()
                    self.speech_frames = 0
                    self.silence_frames = 0
                self.speech_buffer.extend(frame_bytes)
                self.speech_frames += 1
                self.silence_frames = 0
            else:
                if self.is_speaking:
                    self.speech_buffer.extend(frame_bytes)
                    self.silence_frames += 1

                    if self.silence_frames >= self.silence_frames_threshold:
                        # Speech ended
                        if self.speech_frames >= self.min_speech_frames:
                            result = bytes(self.speech_buffer)
                        self.is_speaking = False
                        self.speech_frames = 0
                        self.silence_frames = 0

        return result


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print("=" * 50)
    print("GeniusPro Voice Server starting...")
    print("=" * 50)

    # Load VAD
    print("[VAD] Loading Silero VAD...")
    app.state.vad_ready = True
    print("[VAD] Ready")

    # Load STT
    stt = STTService(model_size=WHISPER_MODEL, device="cuda")
    stt.load()
    app.state.stt = stt

    # Load TTS
    tts = TTSService(model_path=VIBEVOICE_MODEL, device="cuda")
    tts.load()
    app.state.tts = tts

    # Conversation lock (one voice session at a time)
    app.state.voice_lock = asyncio.Lock()

    print("=" * 50)
    print("GeniusPro Voice Server ready!")
    print(f"  STT: Whisper {WHISPER_MODEL}")
    print(f"  Brain: {OLLAMA_MODEL} via Ollama")
    print(f"  TTS: VibeVoice-Realtime-0.5B")
    print(f"  Voice: {VOICE_PRESET}")
    print("=" * 50)


# ─── WebSocket Voice Endpoint ─────────────────────────────────────────────────

@app.websocket("/v1/voice")
async def voice_endpoint(ws: WebSocket):
    await ws.accept()
    print("[voice] Client connected")

    # Send config to client
    await ws.send_text(json.dumps({
        "type": "config",
        "input_sample_rate": INPUT_SAMPLE_RATE,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
    }))

    lock: asyncio.Lock = app.state.voice_lock
    if lock.locked():
        await ws.send_text(json.dumps({"type": "error", "message": "Voice mode busy, try again later"}))
        await ws.close(code=1013, reason="Busy")
        return

    async with lock:
        stt: STTService = app.state.stt
        tts: TTSService = app.state.tts
        vad = VADProcessor()
        conversation_history = []

        # System prompt for voice mode
        conversation_history.append({
            "role": "system",
            "content": (
                "You are GeniusPro Voice Assistant, a helpful AI assistant. "
                "Keep responses concise and conversational since they will be spoken aloud. "
                "Avoid code blocks, markdown, or special formatting -- just speak naturally. "
                "If asked about code, explain it verbally in simple terms."
            ),
        })

        try:
            while True:
                try:
                    data = await ws.receive_bytes()
                except WebSocketDisconnect:
                    print("[voice] Client disconnected")
                    break

                # Feed audio to VAD
                speech_audio = vad.process_chunk(data)

                if speech_audio is None:
                    # Still listening, send status
                    if vad.is_speaking:
                        await ws.send_text(json.dumps({"type": "status", "state": "listening"}))
                    continue

                # Speech detected and ended -- process it
                print(f"[voice] Speech detected: {len(speech_audio)} bytes")
                await ws.send_text(json.dumps({"type": "status", "state": "thinking"}))

                # 1. Transcribe
                text = await asyncio.to_thread(stt.transcribe, speech_audio)
                print(f"[voice] Transcribed: {text}")

                if not text or len(text.strip()) < 2:
                    await ws.send_text(json.dumps({"type": "status", "state": "idle"}))
                    continue

                await ws.send_text(json.dumps({"type": "transcript", "text": text}))

                # 2. Get response from Ollama
                conversation_history.append({"role": "user", "content": text})

                await ws.send_text(json.dumps({"type": "status", "state": "speaking"}))

                full_response = ""
                async for chunk in ollama_chat_stream(conversation_history):
                    full_response += chunk

                conversation_history.append({"role": "assistant", "content": full_response})
                print(f"[voice] Response: {full_response[:100]}...")

                await ws.send_text(json.dumps({"type": "response", "text": full_response}))

                # 3. Generate speech and stream back
                stop_event = threading.Event()

                def generate_audio():
                    return list(tts.stream(full_response, stop_event=stop_event))

                audio_chunks = await asyncio.to_thread(generate_audio)

                for chunk in audio_chunks:
                    if ws.client_state != WebSocketState.CONNECTED:
                        break
                    pcm_bytes = tts.chunk_to_pcm16(chunk)
                    await ws.send_bytes(pcm_bytes)

                # Signal end of audio
                await ws.send_text(json.dumps({"type": "audio_end"}))
                await ws.send_text(json.dumps({"type": "status", "state": "idle"}))

        except Exception as e:
            print(f"[voice] Error: {e}")
            traceback.print_exc()
            try:
                await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass
        finally:
            print("[voice] Session ended")


# ─── Health / Info ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "geniuspro-voice"}


@app.get("/v1/voice/config")
def voice_config():
    tts: TTSService = app.state.tts
    return {
        "voices": sorted(tts.voice_presets.keys()),
        "default_voice": tts.default_voice_key,
        "stt_model": WHISPER_MODEL,
        "llm_model": OLLAMA_MODEL,
        "tts_model": VIBEVOICE_MODEL,
    }


# ─── Web Client ───────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(BASE / "client.html")
