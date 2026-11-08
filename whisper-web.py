# app.py - Main Flask Application
from flask import Flask, request, jsonify, render_template_string
import os
# Must be set before CUDA is initialised: lets empty_cache() return memory even when a few small
# tensors survive, instead of a whole large block staying reserved.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import tempfile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json
import subprocess
import uuid
from werkzeug.utils import secure_filename
import threading
import time
import gc
import traceback
import base64
import glob
import re
import shutil
import urllib.request
import urllib.error
import wave

import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Serialize GPU work: the model is loaded per request, so concurrent requests
# would otherwise each load their own copy and exhaust GPU memory.
gpu_lock = threading.Lock()

# Progress tracking
progress_data = {}

MODEL_ID = "openai/whisper-large-v3"

# Alternative engine: Qwen3-ASR served by Ollama
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.containers.internal:11434").rstrip("/")
QWEN_ASR_MODEL = os.environ.get("QWEN_ASR_MODEL", "frozenlab/qwen3-asr:1.7b")
QWEN_ASR_CHUNK_S = int(os.environ.get("QWEN_ASR_CHUNK_S", "30"))  # model accepts <= ~60 s per request
QWEN_ASR_TIMEOUT_S = int(os.environ.get("QWEN_ASR_TIMEOUT_S", "300"))

# Alternative engine: Microsoft VibeVoice-ASR (transcription + speaker diarization in one model)
VIBEVOICE_MODEL_ID = os.environ.get("VIBEVOICE_MODEL_ID", "microsoft/VibeVoice-ASR-HF")
VIBEVOICE_SAMPLE_RATE = 24000
VIBEVOICE_MAX_PART_S = int(os.environ.get("VIBEVOICE_MAX_PART_S", "2700"))  # 45 min tested on 24 GB (peak 23.2 GB); model limit is 60 min

# GPU memory needed before loading a local model; Ollama models are unloaded to make room if necessary
WHISPER_MIN_FREE_GB = float(os.environ.get("WHISPER_MIN_FREE_GB", "6"))
VIBEVOICE_MIN_FREE_GB = float(os.environ.get("VIBEVOICE_MIN_FREE_GB", "20"))
OLLAMA_UNLOAD_FOR_GPU = os.environ.get("OLLAMA_UNLOAD_FOR_GPU", "true").lower() == "true"

# Alternative engine: Whisper + pyannote speaker diarization (gated model: needs a Hugging Face token)
DIARIZATION_MODEL_ID = os.environ.get("DIARIZATION_MODEL_ID", "pyannote/speaker-diarization-community-1")
WHISPER_PYANNOTE_MIN_FREE_GB = float(os.environ.get("WHISPER_PYANNOTE_MIN_FREE_GB", "8"))
WHISPER_PYANNOTE_BATCH = int(os.environ.get("WHISPER_PYANNOTE_BATCH", "8"))

ENGINES = ("whisper", "qwen3-asr", "vibevoice", "whisper-pyannote")

# Post-processing (minutes / action items) with an LLM served by Ollama
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4:31b")
LLM_MAX_CTX = int(os.environ.get("LLM_MAX_CTX", "32768"))
LLM_TIMEOUT_S = int(os.environ.get("LLM_TIMEOUT_S", "900"))
ANALYSES = ("minutes", "actions")


def load_whisper_model():
    """Load the Whisper model and processor. Called per request; weights come from the local HF cache."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Loading Whisper model on {device}...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Model loaded.")
    return model, processor


def release_gpu_memory():
    """Free unreferenced tensors and return cached GPU memory to the driver."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # The cuBLAS workspace lives in the caching allocator and pins its memory block otherwise
        torch._C._cuda_clearCublasWorkspaces()  # pylint: disable=protected-access
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"GPU memory after release: allocated={torch.cuda.memory_allocated() / 2**20:.0f} MiB, "
              f"reserved={torch.cuda.memory_reserved() / 2**20:.0f} MiB")


def _model_key(name):
    return name if ":" in name.split("/")[-1] else f"{name}:latest"


def unload_ollama_models(keep=None):
    """Unload every model Ollama holds, except `keep`, so only one model uses the GPU at a time.
    Waits until Ollama reports them gone. Call with gpu_lock held."""
    if not OLLAMA_UNLOAD_FOR_GPU:
        return
    try:
        loaded = _ollama_get("/api/ps").get("models", [])
    except RuntimeError as e:
        print(f"Could not query Ollama: {e}")
        return
    to_unload = [m["name"] for m in loaded if not keep or _model_key(m["name"]) != _model_key(keep)]
    for name in to_unload:
        print(f"Unloading Ollama model {name} (one model at a time)")
        _ollama_post("/api/generate", {"model": name, "keep_alive": 0}, 60)
    for _ in range(60):
        if not to_unload:
            return
        still = {m["name"] for m in _ollama_get("/api/ps").get("models", [])}
        to_unload = [n for n in to_unload if n in still]
        time.sleep(1)


def ensure_gpu_memory(min_free_gb, task_id):
    """Unload all Ollama models, then make sure enough GPU memory is free. Call with gpu_lock held."""
    if not torch.cuda.is_available():
        return
    progress_data[task_id] = {"stage": "freeing_gpu", "progress": 38}
    unload_ollama_models()

    for _ in range(60):
        free_gb = torch.cuda.mem_get_info()[0] / 2**30
        if free_gb >= min_free_gb:
            return
        time.sleep(1)
    raise RuntimeError(f"Not enough free GPU memory: {free_gb:.1f} GB free, {min_free_gb:.0f} GB needed")


def load_vibevoice_model():
    from transformers import VibeVoiceAsrForConditionalGeneration  # needs transformers >= 5.3
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Loading VibeVoice-ASR on {device}...")
    processor = AutoProcessor.from_pretrained(VIBEVOICE_MODEL_ID)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(VIBEVOICE_MODEL_ID, dtype=dtype, device_map=device)
    model.eval()
    print("VibeVoice-ASR loaded.")
    return model, processor


def _load_audio_array(audio_file, sample_rate):
    """Decode any audio/video file to a mono float32 numpy array via FFmpeg."""
    command = ["ffmpeg", "-nostdin", "-i", audio_file, "-vn", "-ac", "1", "-ar", str(sample_rate),
               "-f", "f32le", "-"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode('utf-8', 'replace')}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("No audio found in file")
    return audio


def _parse_vibevoice_output(raw):
    """Parse '[{"Start":..,"End":..,"Speaker":..,"Content":..}, ...]', salvaging complete segments if truncated."""
    text = raw.strip()
    if text.startswith("assistant"):
        text = text[len("assistant"):].strip()
    try:
        segments = json.loads(text[text.index("["):])
        if isinstance(segments, list):
            return [seg for seg in segments if isinstance(seg, dict) and "Content" in seg]
    except ValueError:
        pass
    segments = []
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            seg = json.loads(match.group(0))
        except ValueError:
            continue
        if isinstance(seg, dict) and "Content" in seg:
            segments.append(seg)
    if not segments and text:
        print(f"Unparseable VibeVoice output: {text[:500]}")
    return segments


def transcribe_vibevoice(audio_file, simple_mode, task_id, model, processor, context=""):
    """Transcribe with speaker labels. Audio longer than VIBEVOICE_MAX_PART_S is processed in parts."""
    audio = _load_audio_array(audio_file, VIBEVOICE_SAMPLE_RATE)
    part_len = VIBEVOICE_MAX_PART_S * VIBEVOICE_SAMPLE_RATE
    parts = [audio[i:i + part_len] for i in range(0, len(audio), part_len)]

    chunks = []
    for p, part in enumerate(parts):
        progress_data[task_id] = {"stage": "transcribing", "progress": 50 + int(45 * p / len(parts)),
                                  "message": f"Teil {p + 1}/{len(parts)}" if len(parts) > 1 else ""}
        duration = len(part) / VIBEVOICE_SAMPLE_RATE
        offset = p * VIBEVOICE_MAX_PART_S
        inputs = processor.apply_transcription_request(audio=part, prompt=context or None)
        inputs = inputs.to(model.device, model.dtype)
        # ~3-4 text tokens per second of speech plus JSON overhead per segment
        max_new_tokens = min(40000, int(duration * 10) + 512)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        raw = processor.decode(generated, skip_special_tokens=True)[0]
        del inputs, output_ids, generated

        for seg in _parse_vibevoice_output(raw):
            content = str(seg.get("Content", "")).strip()
            if not content:
                continue
            spk = int(seg.get("Speaker", 0)) + 1
            # Speaker numbers are only consistent within one part
            speaker = f"Sprecher {spk}" if len(parts) == 1 else f"Sprecher {p + 1}.{spk}"
            start = float(seg.get("Start", 0) or 0) + offset
            end = float(seg.get("End", start - offset) or 0) + offset
            chunks.append({"timestamp": [round(start, 2), round(end, 2)], "speaker": speaker, "text": content})

    progress_data[task_id] = {"stage": "completed", "progress": 100}
    return _speaker_result(chunks, simple_mode)


def _speaker_result(chunks, simple_mode):
    """Result format for engines with speaker labels ("Sprecher N" or "Sprecher P.N")."""
    speakers = sorted({c["speaker"] for c in chunks}, key=lambda x: [float(n) for n in x.split()[1].split(".")])
    text = "\n".join(f"{c['speaker']}: {c['text']}" for c in chunks)
    result = {"text": text, "speakers": speakers}
    if not simple_mode:
        result["chunks"] = chunks
    return result


def load_diarization_pipeline():
    from pyannote.audio import Pipeline  # imported lazily: heavy import, only needed for this engine
    print(f"Loading diarization pipeline {DIARIZATION_MODEL_ID}...")
    try:
        diarization = Pipeline.from_pretrained(DIARIZATION_MODEL_ID, token=os.environ.get("HF_TOKEN"))
    except Exception as e:  # pylint: disable=broad-except
        raise RuntimeError(
            f"Could not load {DIARIZATION_MODEL_ID}: {e}. The model is gated: accept its terms on "
            f"https://huggingface.co/{DIARIZATION_MODEL_ID} and provide a Hugging Face token "
            "(HF_TOKEN or a token file in the mounted Hugging Face cache)."
        ) from None
    if diarization is None:
        raise RuntimeError(f"Could not load {DIARIZATION_MODEL_ID}: access denied or missing token")
    if torch.cuda.is_available():
        diarization.to(torch.device("cuda:0"))
    return diarization


def _diarize(diarization, audio):
    """Return sorted (start, end, label) turns without overlaps, labels renamed to 'Sprecher N'."""
    output = diarization({"waveform": torch.from_numpy(audio.copy()).unsqueeze(0), "sample_rate": 16000})
    # pyannote 4 returns an object with an overlap-free variant, which is what transcript alignment needs
    annotation = getattr(output, "exclusive_speaker_diarization", None) or getattr(output, "speaker_diarization", output)
    turns, names = [], {}
    for turn, _, label in sorted(annotation.itertracks(yield_label=True), key=lambda t: t[0].start):
        if label not in names:
            names[label] = f"Sprecher {len(names) + 1}"
        turns.append((turn.start, turn.end, names[label]))
    return turns


def _turn_pieces(turns, max_len=28.0, max_gap=1.0, min_len=0.3):
    """Merge consecutive turns of the same speaker and cut them into Whisper-sized pieces (<= max_len seconds)."""
    merged = []
    for start, end, speaker in turns:
        last = merged[-1] if merged else None
        if last and last[2] == speaker and start - last[1] <= max_gap and end - last[0] <= max_len:
            last[1] = end
        else:
            merged.append([start, end, speaker])
    pieces = []
    for start, end, speaker in merged:
        while end - start > max_len:
            pieces.append((start, start + max_len, speaker))
            start += max_len
        if end - start >= min_len:
            pieces.append((start, end, speaker))
    return pieces


def transcribe_whisper_pyannote(audio_file, simple_mode, task_id, model, processor, diarization):
    """pyannote finds the speaker turns; Whisper transcribes the turn pieces in batches (as WhisperX does)."""
    audio = _load_audio_array(audio_file, 16000)

    progress_data[task_id] = {"stage": "diarizing", "progress": 45}
    pieces = _turn_pieces(_diarize(diarization, audio))

    progress_data[task_id] = {"stage": "transcribing", "progress": 60}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
    )
    inputs = [{"raw": audio[int(a * 16000):int(b * 16000)], "sampling_rate": 16000} for a, b, _ in pieces]
    # Language is detected per piece, which handles meetings switching between German and English
    outputs = pipe(inputs, batch_size=WHISPER_PYANNOTE_BATCH, generate_kwargs={"task": "transcribe"}) if inputs else []
    del pipe

    chunks = []
    for (start, end, speaker), out in zip(pieces, outputs):
        text = out.get("text", "").strip()
        if not text:
            continue
        last = chunks[-1] if chunks else None
        if last and last["speaker"] == speaker:
            last["text"] += " " + text
            last["timestamp"][1] = round(end, 2)
        else:
            chunks.append({"timestamp": [round(start, 2), round(end, 2)], "speaker": speaker, "text": text})

    progress_data[task_id] = {"stage": "completed", "progress": 100}
    return _speaker_result(chunks, simple_mode)


def run_transcription(audio_file, simple_mode, task_id, engine="whisper", context=""):
    """Load the model, transcribe, and unload the model again - even on failure."""
    progress_data[task_id] = {"stage": "queued", "progress": 35}
    with gpu_lock:
        model = processor = diarization = None
        result = None
        error_msg = None
        try:
            if engine == "vibevoice":
                ensure_gpu_memory(VIBEVOICE_MIN_FREE_GB, task_id)
                progress_data[task_id] = {"stage": "loading_model", "progress": 40}
                model, processor = load_vibevoice_model()
                result = transcribe_vibevoice(audio_file, simple_mode, task_id, model, processor, context)
            elif engine == "whisper-pyannote":
                ensure_gpu_memory(WHISPER_PYANNOTE_MIN_FREE_GB, task_id)
                progress_data[task_id] = {"stage": "loading_model", "progress": 40}
                diarization = load_diarization_pipeline()
                model, processor = load_whisper_model()
                result = transcribe_whisper_pyannote(audio_file, simple_mode, task_id, model, processor,
                                                     diarization)
            else:
                ensure_gpu_memory(WHISPER_MIN_FREE_GB, task_id)
                progress_data[task_id] = {"stage": "loading_model", "progress": 40}
                model, processor = load_whisper_model()
                result = transcribe_audio(audio_file, simple_mode, task_id, model, processor)
        except Exception as e:  # pylint: disable=broad-except
            # Keep only the message: holding the exception/traceback would keep
            # the model referenced via stack frames and block the memory release.
            traceback.print_exc()
            error_msg = f"{type(e).__name__}: {e}"
        finally:
            model = processor = diarization = None
            release_gpu_memory()

    if error_msg is not None:
        raise RuntimeError(error_msg)
    return result

def extract_audio(video_file, audio_file, task_id):
    """Extract audio from video file using FFmpeg"""
    progress_data[task_id] = {"stage": "extracting", "progress": 10}

    command = ["ffmpeg", "-i", video_file, "-map", "0:a", audio_file, "-y"]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"FFmpeg error: {result.stderr}")

    progress_data[task_id] = {"stage": "extracted", "progress": 30}

def transcribe_audio(audio_file, simple_mode, task_id, model, processor):
    """Transcribe audio file using Whisper"""

    progress_data[task_id] = {"stage": "transcribing", "progress": 50}
    
    # Get audio file size
    file_size = os.path.getsize(audio_file) / (1024 * 1024)  # Size in MB
    
    # For large files, use more conservative settings
    chunk_length = 30
    batch_size = 8
    if file_size > 50:  # If file is larger than 50MB
        chunk_length = 15
        batch_size = 4
    elif file_size > 100:  # If file is larger than 100MB
        chunk_length = 10
        batch_size = 2
        
    # Clear GPU cache before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print("starting cormal operation")
        # Set generation parameters
        generate_kwargs = {
        #    "language": "<|en|>", 
            "task": "transcribe"
            # Let the pipeline handle attention_mask automatically
        }
        
        print("simple mode: ", simple_mode)
        if simple_mode:
            # Create a new pipeline with dynamic settings for simple mode
            pipe_simple = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=chunk_length,
                batch_size=batch_size,
                return_timestamps=False,
                torch_dtype=torch_dtype,
                device=device,
            )
            result = pipe_simple(audio_file, generate_kwargs=generate_kwargs)
        else:
            # Create a new pipeline with timestamps and dynamic settings
            pipe_timestamps = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=chunk_length,
                batch_size=batch_size,
                return_timestamps=True,
                torch_dtype=torch_dtype,
                device=device,
            )
            result = pipe_timestamps(audio_file, generate_kwargs=generate_kwargs)

    except ValueError as e:
        # Log the error
        print(f"Pipeline error, trying fallback: {str(e)}")
        
        # Clear GPU cache before fallback
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            # Use even more conservative settings for fallback
            fallback_chunk_length = max(5, chunk_length // 2)
            fallback_batch_size = max(1, batch_size // 2)
            
            # Create a fallback pipeline with minimal settings
            pipe_fallback = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=64,  # Reduced tokens
                chunk_length_s=fallback_chunk_length,
                batch_size=fallback_batch_size,
                return_timestamps=not simple_mode,
                torch_dtype=torch.float32,  # Try with float32 for better stability
                device=device,
            )
            # Use the same generate_kwargs without attention_mask
            result = pipe_fallback(audio_file, generate_kwargs=generate_kwargs)
        except Exception as fallback_error:
            # If fallback also fails, log and raise a more specific error
            print(f"Fallback also failed: {str(fallback_error)}")
            raise RuntimeError(f"Transcription failed: {str(e)}. Fallback also failed: {str(fallback_error)}")

    progress_data[task_id] = {"stage": "completed", "progress": 100}
    return result

def _clean_qwen_asr_text(text):
    """Qwen3-ASR prefixes its answer with 'language <Lang><asr_text>'; keep only the transcript."""
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    text = re.sub(r"</?asr_text>", "", text)
    return text.strip()


def _ollama_transcribe_chunk(wav_path):
    with open(wav_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")
    payload = {
        "model": QWEN_ASR_MODEL,
        "messages": [{"role": "user", "content": "", "images": [audio_b64]}],
        "stream": False,
        "options": {"temperature": 0},
    }
    body = _ollama_post("/api/chat", payload, QWEN_ASR_TIMEOUT_S)
    return _clean_qwen_asr_text(body.get("message", {}).get("content", ""))


def transcribe_with_ollama(audio_file, simple_mode, task_id):
    """Transcribe with Qwen3-ASR via Ollama: split into 16 kHz mono WAV chunks and send them one by one."""
    progress_data[task_id] = {"stage": "transcribing", "progress": 40}
    chunk_dir = tempfile.mkdtemp(prefix="qwen_asr_")
    try:
        command = ["ffmpeg", "-i", audio_file, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                   "-f", "segment", "-segment_time", str(QWEN_ASR_CHUNK_S),
                   os.path.join(chunk_dir, "chunk_%05d.wav"), "-y"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

        chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.wav")))
        if not chunk_files:
            raise RuntimeError("No audio found in file")

        chunks = []
        offset = 0.0
        for i, chunk_file in enumerate(chunk_files):
            with wave.open(chunk_file, "rb") as w:
                duration = w.getnframes() / float(w.getframerate())
            text = _ollama_transcribe_chunk(chunk_file)
            if text:
                chunks.append({"timestamp": [round(offset, 2), round(offset + duration, 2)], "text": text})
            offset += duration
            progress_data[task_id] = {"stage": "transcribing",
                                      "progress": 40 + int(55 * (i + 1) / len(chunk_files))}
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)

    progress_data[task_id] = {"stage": "completed", "progress": 100}
    full_text = " ".join(c["text"] for c in chunks)
    if simple_mode:
        return {"text": full_text}
    return {"text": full_text, "chunks": chunks}


def _ollama_post(path, payload, timeout):
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama error {e.code}: {e.read().decode('utf-8', 'replace')}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_URL}: {e.reason}") from None


def _ollama_get(path):
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}{path}", timeout=10) as resp:
            return json.load(resp)
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_URL}: {e}") from None


def _estimate_tokens(text):
    return len(text) // 3 + 1  # conservative for German


def llm_chat(system, user, max_output_tokens=4096):
    """Single non-streaming chat call; context size sized to the input."""
    needed = _estimate_tokens(system) + _estimate_tokens(user) + max_output_tokens
    num_ctx = min(LLM_MAX_CTX, max(8192, 1 << (needed - 1).bit_length()))
    body = _ollama_post("/api/chat", {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_ctx": num_ctx, "num_predict": max_output_tokens},
    }, LLM_TIMEOUT_S)
    text = body.get("message", {}).get("content", "")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _split_text(text, max_chars):
    """Split at sentence boundaries into parts of at most ~max_chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


TRANSCRIPT_NOTE = (
    "Das Transkript stammt aus einer automatischen Spracherkennung und kann Erkennungsfehler enthalten. "
    "Es kann Deutsch und Englisch gemischt sein. Beginnen Zeilen mit einem Sprecher (z. B. \"Sprecher 2:\" "
    "oder einem Namen), ordne Aussagen, Zusagen und Aufgaben anhand dieser Kennzeichnung zu; sonst gibt es "
    "keine Sprecherkennzeichnung. Verwende ausschließlich Informationen aus dem Transkript, erfinde nichts. "
    "Antworte auf Deutsch in Markdown."
)

MINUTES_SYSTEM = (
    "Du bist ein erfahrener Protokollführer und erstellst aus Besprechungstranskripten ein Ergebnisprotokoll. "
    + TRANSCRIPT_NOTE + "\n\nGliederung:\n"
    "# Protokoll: <Thema>\n"
    "**Datum:** (nur falls genannt) · **Teilnehmende:** (nur falls erkennbar)\n"
    "## Zusammenfassung\n(3–5 Sätze)\n"
    "## Besprochene Themen\n(je Thema eine Zwischenüberschrift mit den wesentlichen Punkten)\n"
    "## Entscheidungen\n"
    "## Aufgaben\n(Tabelle: Aufgabe | Verantwortlich | Frist – unbekannt bleibt leer)\n"
    "## Offene Fragen\n"
    "Lasse Abschnitte ohne Inhalt weg."
)


def _actions_system(name):
    if name:
        who = (f"Liste ausschließlich die Aufgaben auf, die {name} übernehmen soll, zugesagt hat oder die {name} "
               f"zugewiesen wurden – auch wenn {name} direkt angesprochen wird (\"kannst du …\") oder in der "
               "Ich-Form zusagt und der Kontext auf diese Person hindeutet. Ist die Zuordnung unsicher, "
               "kennzeichne den Punkt mit \"(Zuordnung unsicher)\".")
    else:
        who = "Liste alle Aufgaben auf und nenne jeweils die verantwortliche Person, falls erkennbar."
    return (
        "Du extrahierst Aufgaben (Action Items) aus Besprechungstranskripten. " + TRANSCRIPT_NOTE + " " + who +
        "\n\nFormat je Aufgabe:\n- [ ] **<Aufgabe als konkrete Handlung>** — Frist: <falls genannt> — "
        "Kontext: <kurzer Bezug zur Besprechung>\n"
        "Gibt es keine passenden Aufgaben, antworte genau mit: \"Keine Aufgaben gefunden.\""
    )


def _analysis_prompt(kind, name):
    if kind == "minutes":
        return MINUTES_SYSTEM, "Erstelle das Protokoll zu folgendem Transkript:"
    return _actions_system(name), "Extrahiere die Aufgaben aus folgendem Transkript:"


def analyze_transcript(transcript, kind, name, task_id):
    """Create meeting minutes or an action item list. Long transcripts are condensed part by part first."""
    system, instruction = _analysis_prompt(kind, name)
    budget_chars = (LLM_MAX_CTX - 6000) * 3

    if len(transcript) > budget_chars:
        parts = _split_text(transcript, budget_chars)
        notes = []
        for i, part in enumerate(parts):
            progress_data[task_id] = {"stage": "analyzing", "progress": 10 + int(70 * i / len(parts)),
                                      "message": f"Abschnitt {i + 1}/{len(parts)}"}
            notes.append(llm_chat(
                "Du fasst einen Abschnitt eines Besprechungstranskripts als ausführliche Stichpunkte zusammen: "
                "Themen, Fakten und Zahlen, Entscheidungen, Aufgaben (mit Person und Frist, wörtlich wie genannt), "
                "offene Fragen. " + TRANSCRIPT_NOTE,
                f"Abschnitt {i + 1} von {len(parts)}:\n\n{part}",
            ))
        transcript = "\n\n".join(f"### Notizen zu Abschnitt {i + 1}\n{n}" for i, n in enumerate(notes))
        instruction = instruction.replace("folgendem Transkript", "folgenden Notizen zu einem Transkript")

    progress_data[task_id] = {"stage": "analyzing", "progress": 85}
    return llm_chat(system, f"{instruction}\n\n{transcript}")


def process_analysis_async(transcript, kind, name, task_id):
    try:
        progress_data[task_id] = {"stage": "queued", "progress": 5}
        with gpu_lock:
            progress_data[task_id] = {"stage": "freeing_gpu", "progress": 8}
            unload_ollama_models(keep=LLM_MODEL)  # e.g. Qwen3-ASR still loaded from the transcription
            progress_data[task_id] = {"stage": "analyzing", "progress": 10}
            text = analyze_transcript(transcript, kind, name, task_id)
        progress_data[task_id] = {"stage": "completed", "progress": 100, "result": {"kind": kind, "markdown": text}}
    except Exception as e:  # pylint: disable=broad-except
        traceback.print_exc()
        progress_data[task_id] = {"stage": "error", "progress": 0, "error": f"Analyse fehlgeschlagen: {e}"}


def process_file_async(file_path, simple_mode, task_id, engine="whisper", context=""):
    """Process file asynchronously"""
    audio_file = None
    try:
        # Check file size first
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"Processing file of size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 500:  # Warn for very large files
            print(f"Warning: Large file ({file_size_mb:.2f} MB) may require significant resources")
            # Update progress to indicate large file processing
            progress_data[task_id] = {"stage": "preparing", "progress": 5, "message": "Large file detected, optimizing processing..."}
            
        # Clear memory before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Check file extension to handle different audio formats
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Special handling for AIFC/AIFF files
        if file_ext in ['.aifc', '.aiff']:
            # Convert AIFC/AIFF to WAV for better compatibility
            audio_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
            progress_data[task_id] = {"stage": "extracting", "progress": 10, "message": "Converting AIFC/AIFF format..."}
            
            try:
                # Use special extraction for AIFC files with error handling
                command = ["ffmpeg", "-i", file_path, "-acodec", "pcm_s16le", "-ar", "16000", audio_file, "-y"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=300)  # 5 minute timeout
                if result.returncode != 0:
                    error_msg = f"FFmpeg error: {result.stderr}"
                    print(error_msg)
                    raise RuntimeError(error_msg)
                progress_data[task_id] = {"stage": "extracted", "progress": 30}
            except subprocess.TimeoutExpired:
                raise RuntimeError("FFmpeg timed out while processing the audio file")
        else:
            # For other files, use standard extraction
            audio_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
            extract_audio(file_path, audio_file, task_id)

        # Transcribe
        if engine == "qwen3-asr":
            # Share the lock with Whisper so a running Whisper job doesn't starve Ollama of GPU memory
            progress_data[task_id] = {"stage": "queued", "progress": 35}
            with gpu_lock:
                progress_data[task_id] = {"stage": "freeing_gpu", "progress": 38}
                unload_ollama_models(keep=QWEN_ASR_MODEL)  # e.g. the LLM from a previous analysis
                result = transcribe_with_ollama(audio_file, simple_mode, task_id)
        else:
            result = run_transcription(audio_file, simple_mode, task_id, engine, context)

        # Clean up
        if audio_file and os.path.exists(audio_file) and audio_file != file_path:
            os.remove(audio_file)
        if os.path.exists(file_path):
            os.remove(file_path)

        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Store result
        progress_data[task_id]["result"] = result

    except (RuntimeError, ValueError, OSError) as e:
        error_msg = f"Error processing file: {str(e)}"
        print(error_msg)
        progress_data[task_id] = {"stage": "error", "progress": 0, "error": error_msg}
        
        # Cleanup on error
        if audio_file and os.path.exists(audio_file) and audio_file != file_path:
            try:
                os.remove(audio_file)
            except OSError:
                pass
                
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    #except Exception as e:
    #    # Catch any other unexpected errors
    #    error_msg = f"Unexpected error: {str(e)}"
    #    print(error_msg)
    #    progress_data[task_id] = {"stage": "error", "progress": 0, "error": error_msg}
    #    
    #    # Cleanup on error
    #    if audio_file and os.path.exists(audio_file) and audio_file != file_path:
    #        try:
    #            os.remove(audio_file)
    #        except OSError:
    #            pass
    #            
    #    if torch.cuda.is_available():
    #        torch.cuda.empty_cache()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    simple_mode = request.form.get('simple', 'false').lower() == 'true'
    engine = request.form.get('engine', 'whisper')
    context = (request.form.get('context') or '').strip()[:2000]
    if engine not in ENGINES:
        return jsonify({'error': f'Unknown engine: {engine}'}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_{filename}")
    file.save(file_path)

    # Generate task ID
    task_id = str(uuid.uuid4())
    progress_data[task_id] = {"stage": "uploaded", "progress": 5}

    # Start processing in background
    thread = threading.Thread(target=process_file_async, args=(file_path, simple_mode, task_id, engine, context))
    thread.daemon = True
    thread.start()

    return jsonify({'task_id': task_id})

@app.route('/analyze', methods=['POST'])
def analyze():
    """Start creating minutes or action items from a transcript"""
    data = request.get_json(silent=True) or {}
    transcript = (data.get('transcript') or '').strip()
    kind = data.get('kind')
    name = (data.get('name') or '').strip()[:100]
    if not transcript:
        return jsonify({'error': 'No transcript provided'}), 400
    if kind not in ANALYSES:
        return jsonify({'error': f'Unknown analysis: {kind}'}), 400

    task_id = str(uuid.uuid4())
    progress_data[task_id] = {"stage": "queued", "progress": 5}
    thread = threading.Thread(target=process_analysis_async, args=(transcript, kind, name, task_id))
    thread.daemon = True
    thread.start()
    return jsonify({'task_id': task_id})

@app.route('/progress/<task_id>')
def get_progress(task_id):
    """Get processing progress"""
    if task_id not in progress_data:
        return jsonify({'error': 'Task not found'}), 404

    return jsonify(progress_data[task_id])

@app.route('/result/<task_id>')
def get_result(task_id):
    """Get transcription result"""
    if task_id not in progress_data:
        return jsonify({'error': 'Task not found'}), 404

    data = progress_data[task_id]
    if 'result' not in data:
        return jsonify({'error': 'Result not ready'}), 202

    result = data['result']
    # Clean up
    del progress_data[task_id]

    return jsonify(result)

# HTML Template embedded in Python
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whisper Transkription</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            padding: 40px;
            backdrop-filter: blur(10px);
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
            font-weight: 300;
        }

        .upload-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            border: 2px dashed #007bff;
            transition: all 0.3s ease;
        }

        .upload-section:hover {
            border-color: #0056b3;
            background: #e3f2fd;
        }

        .file-input-wrapper {
            position: relative;
            display: inline-block;
            width: 100%;
        }

        .file-input {
            position: absolute;
            left: -9999px;
        }

        .file-input-button {
            display: inline-block;
            padding: 15px 30px;
            background: linear-gradient(45deg, #007bff, #0056b3);
            color: white;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 16px;
            border: none;
            width: 100%;
            text-align: center;
        }

        .file-input-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 123, 255, 0.3);
        }

        .options {
            display: flex;
            align-items: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }

        .checkbox-wrapper {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .process-button {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 25px;
            font-size: 18px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
            width: 100%;
        }

        .process-button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(40, 167, 69, 0.3);
        }

        .process-button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }

        .status {
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: 500;
        }

        .status.processing {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }

        .status.success {
            background: #d1edff;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }

        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }

        .results {
            margin-top: 30px;
        }

        .transcript-container {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin-top: 20px;
            border-left: 5px solid #007bff;
        }

        .timestamp-chunk {
            margin-bottom: 15px;
            padding: 10px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        .timestamp {
            font-size: 12px;
            color: #6c757d;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .text-content {
            line-height: 1.6;
            color: #333;
        }

        .simple-transcript {
            line-height: 1.8;
            color: #333;
            font-size: 16px;
        }

        .file-info {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 4px solid #2196f3;
        }

        .progress-bar {
            width: 100%;
            height: 6px;
            background: #e9ecef;
            border-radius: 3px;
            overflow: hidden;
            margin: 10px 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #007bff, #0056b3);
            width: 0%;
            transition: width 0.3s ease;
        }

        .download-button {
            background: linear-gradient(45deg, #17a2b8, #138496);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 15px;
        }

        .context-input {
            padding: 8px 12px;
            border: 1px solid #ced4da;
            border-radius: 8px;
            font-size: 1em;
            min-width: 320px;
        }

        .speaker {
            font-weight: 600;
            color: #6f42c1;
            margin-right: 6px;
        }

        .speaker-names {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 10px 0 5px;
        }

        .speaker-names label {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .speaker-names input {
            padding: 6px 10px;
            border: 1px solid #ced4da;
            border-radius: 6px;
            width: 160px;
        }

        .simple-transcript {
            white-space: pre-wrap;
        }

        .analysis-section {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
        }

        .analysis-controls {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin: 15px 0;
        }

        .analysis-controls input {
            padding: 10px 14px;
            border: 1px solid #ced4da;
            border-radius: 8px;
            font-size: 1em;
            min-width: 220px;
        }

        .analysis-output {
            white-space: pre-wrap;
            background: #fff;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            line-height: 1.5;
            font-family: inherit;
        }

        .download-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 5px 15px rgba(23, 162, 184, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Whisper Transkription v4</h1>

        <div class="upload-section">
            <div class="file-input-wrapper">
                <input type="file" id="fileInput" class="file-input" accept="audio/*,video/*,.aifc,.aiff">
                <label for="fileInput" class="file-input-button">
                    📁 Audio/Video Datei auswählen
                </label>
            </div>

            <div id="fileInfo" class="file-info" style="display: none;"></div>

            <div class="options">
                <div class="checkbox-wrapper">
                    <label for="engine">Modell:</label>
                    <select id="engine">
                        <option value="whisper" selected>Whisper large-v3</option>
                        <option value="qwen3-asr">Qwen3-ASR 1.7B (Ollama)</option>
                        <option value="vibevoice">VibeVoice-ASR (mit Sprechererkennung)</option>
                        <option value="whisper-pyannote">Whisper + pyannote (mit Sprechererkennung, schneller)</option>
                    </select>
                </div>
                <div class="checkbox-wrapper" id="contextWrapper" style="display: none;">
                    <label for="context">Kontext:</label>
                    <input type="text" id="context" class="context-input"
                           placeholder="Namen, Fachbegriffe, Thema (optional)">
                </div>
                <div class="checkbox-wrapper">
                    <input type="checkbox" id="simpleMode" checked>
                    <label for="simpleMode">Einfacher Modus (ohne Zeitstempel)</label>
                </div>
            </div>

            <button id="processButton" class="process-button" disabled>
                🚀 Transkription starten
            </button>
        </div>

        <div id="status" class="status" style="display: none;"></div>

        <div class="progress-bar" id="progressBar" style="display: none;">
            <div class="progress-fill" id="progressFill"></div>
        </div>

        <div id="results" class="results" style="display: none;">
            <h2>📝 Transkript</h2>
            <div id="transcriptContainer" class="transcript-container"></div>
            <button id="downloadButton" class="download-button" style="display: none;">
                💾 Als JSON herunterladen
            </button>

            <div class="analysis-section">
                <h2>🧠 Auswertung (Ollama)</h2>
                <div id="speakerNamesWrapper" style="display: none;">
                    <p>Sprecher benennen (wird für Protokoll und Aufgaben verwendet):</p>
                    <div id="speakerNames" class="speaker-names"></div>
                </div>
                <div class="analysis-controls">
                    <input type="text" id="userName" placeholder="Ihr Name (für Ihre Aufgaben)">
                    <button id="minutesButton" class="download-button">📋 Protokoll erstellen</button>
                    <button id="actionsButton" class="download-button">✅ Meine Aufgaben</button>
                </div>
                <div id="analysisStatus" class="status" style="display: none;"></div>
                <div id="analysisResult" style="display: none;">
                    <h3 id="analysisTitle"></h3>
                    <div id="analysisOutput" class="analysis-output"></div>
                    <button id="analysisCopyButton" class="download-button">📄 Kopieren</button>
                    <button id="analysisDownloadButton" class="download-button">💾 Als Markdown herunterladen</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        class WhisperApp {
            constructor() {
                this.currentFile = null;
                this.currentTask = null;
                this.transcriptionResult = null;
                this.initializeEventListeners();
            }

            initializeEventListeners() {
                const fileInput = document.getElementById('fileInput');
                const processButton = document.getElementById('processButton');
                const downloadButton = document.getElementById('downloadButton');

                fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
                const engine = document.getElementById('engine');
                const toggleContext = () => {
                    document.getElementById('contextWrapper').style.display =
                        engine.value === 'vibevoice' ? 'flex' : 'none';
                };
                engine.addEventListener('change', toggleContext);
                toggleContext();
                processButton.addEventListener('click', () => this.processFile());
                downloadButton.addEventListener('click', () => this.downloadResult());

                const userName = document.getElementById('userName');
                try { userName.value = localStorage.getItem('whisperUserName') || ''; } catch (e) {}
                userName.addEventListener('change', () => {
                    try { localStorage.setItem('whisperUserName', userName.value.trim()); } catch (e) {}
                });
                document.getElementById('minutesButton').addEventListener('click', () => this.runAnalysis('minutes'));
                document.getElementById('actionsButton').addEventListener('click', () => this.runAnalysis('actions'));
                document.getElementById('analysisCopyButton').addEventListener('click', () => {
                    navigator.clipboard.writeText(this.analysisMarkdown || '');
                });
                document.getElementById('analysisDownloadButton').addEventListener('click', () => this.downloadAnalysis());
            }

            handleFileSelect(event) {
                const file = event.target.files[0];
                if (!file) return;

                this.currentFile = file;
                const fileInfo = document.getElementById('fileInfo');
                const processButton = document.getElementById('processButton');

                fileInfo.innerHTML = `
                    <strong>Ausgewählte Datei:</strong> ${file.name}<br>
                    <strong>Größe:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB<br>
                    <strong>Typ:</strong> ${file.type}
                `;
                fileInfo.style.display = 'block';
                processButton.disabled = false;

                document.getElementById('results').style.display = 'none';
                document.getElementById('status').style.display = 'none';
            }

            async processFile() {
                if (!this.currentFile) return;

                const formData = new FormData();
                formData.append('file', this.currentFile);
                formData.append('simple', document.getElementById('simpleMode').checked);
                formData.append('engine', document.getElementById('engine').value);
                formData.append('context', document.getElementById('context').value);

                try {
                    this.showStatus('Upload startet...', 'processing');

                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`Upload failed: ${response.statusText}`);
                    }

                    const data = await response.json();
                    this.currentTask = data.task_id;

                    this.showProgress();
                    this.pollProgress();

                } catch (error) {
                    this.showStatus(`Fehler: ${error.message}`, 'error');
                    this.hideProgress();
                }
            }

            async pollProgress() {
                if (!this.currentTask) return;

                try {
                    const response = await fetch(`/progress/${this.currentTask}`);
                    const data = await response.json();

                    if (data.error) {
                        throw new Error(data.error);
                    }

                    const progressFill = document.getElementById('progressFill');
                    progressFill.style.width = data.progress + '%';

                    const statusMessages = {
                        'uploaded': 'Datei hochgeladen...',
                        'extracting': 'Audio wird extrahiert...',
                        'extracted': 'Audio extrahiert...',
                        'queued': 'Warte auf freie GPU...',
                        'loading_model': 'Modell wird geladen...',
                        'diarizing': 'Sprecher werden erkannt...',
                        'freeing_gpu': 'GPU-Speicher wird freigegeben (Ollama-Modelle entladen)...',
                        'transcribing': 'Transkription läuft...',
                        'completed': 'Verarbeitung abgeschlossen!',
                        'error': 'Fehler aufgetreten'
                    };

                    this.showStatus(statusMessages[data.stage] || 'Verarbeitung...', 'processing');

                    if (data.stage === 'completed') {
                        await this.fetchResult();
                    } else if (data.stage === 'error') {
                        throw new Error(data.error || 'Unbekannter Fehler');
                    } else {
                        setTimeout(() => this.pollProgress(), 2000);
                    }

                } catch (error) {
                    this.showStatus(`Fehler: ${error.message}`, 'error');
                    this.hideProgress();
                }
            }

            async fetchResult() {
                try {
                    const response = await fetch(`/result/${this.currentTask}`);
                    const data = await response.json();

                    if (data.error) {
                        throw new Error(data.error);
                    }

                    this.transcriptionResult = data;
                    this.showStatus('Transkription erfolgreich abgeschlossen!', 'success');
                    this.hideProgress();
                    this.displayResults();

                } catch (error) {
                    this.showStatus(`Fehler beim Abrufen des Ergebnisses: ${error.message}`, 'error');
                    this.hideProgress();
                }
            }

            displayResults() {
                const results = document.getElementById('results');
                const container = document.getElementById('transcriptContainer');
                const downloadButton = document.getElementById('downloadButton');

                const esc = (t) => String(t).replace(/[&<>"']/g, c => ({
                    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
                })[c]);
                if (this.transcriptionResult.chunks) {
                    container.innerHTML = this.transcriptionResult.chunks.map(chunk => `
                        <div class="timestamp-chunk">
                            <div class="timestamp">
                                ${this.formatTime(chunk.timestamp[0])} - ${this.formatTime(chunk.timestamp[1])}
                            </div>
                            <div class="text-content">${chunk.speaker
                                ? `<span class="speaker" data-speaker="${esc(chunk.speaker)}">${esc(chunk.speaker)}:</span>`
                                : ''}${esc(chunk.text)}</div>
                        </div>
                    `).join('');
                } else {
                    container.innerHTML = `<div class="simple-transcript">${esc(this.transcriptionResult.text)}</div>`;
                }

                const speakers = this.transcriptionResult.speakers || [];
                const namesBox = document.getElementById('speakerNames');
                namesBox.innerHTML = speakers.map(sp => `
                    <label>${esc(sp)} = <input type="text" data-speaker="${esc(sp)}" placeholder="Name"></label>
                `).join('');
                namesBox.querySelectorAll('input').forEach(input => {
                    input.addEventListener('input', () => {
                        container.querySelectorAll(`.speaker[data-speaker="${input.dataset.speaker}"]`)
                            .forEach(el => el.textContent = (input.value.trim() || input.dataset.speaker) + ':');
                    });
                });
                document.getElementById('speakerNamesWrapper').style.display = speakers.length ? 'block' : 'none';

                results.style.display = 'block';
                downloadButton.style.display = 'inline-block';
            }

            async runAnalysis(kind) {
                if (!this.transcriptionResult) return;
                const name = document.getElementById('userName').value.trim();
                const status = document.getElementById('analysisStatus');
                const buttons = [document.getElementById('minutesButton'), document.getElementById('actionsButton')];
                const setStatus = (msg, type) => {
                    status.textContent = msg;
                    status.className = `status ${type}`;
                    status.style.display = 'block';
                };

                buttons.forEach(b => b.disabled = true);
                setStatus('Auswertung wird gestartet...', 'processing');
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({transcript: this.namedTranscript(), kind, name})
                    });
                    const start = await response.json();
                    if (!response.ok || start.error) throw new Error(start.error || response.statusText);

                    const messages = {
                        'queued': 'Warte auf freie GPU...',
                        'freeing_gpu': 'GPU-Speicher wird freigegeben...',
                        'analyzing': 'Sprachmodell arbeitet (kann einige Minuten dauern)...'
                    };
                    while (true) {
                        await new Promise(r => setTimeout(r, 2000));
                        const data = await (await fetch(`/progress/${start.task_id}`)).json();
                        if (data.stage === 'error' || data.error) throw new Error(data.error || 'Unbekannter Fehler');
                        if (data.stage === 'completed') break;
                        setStatus((messages[data.stage] || 'Verarbeitung...') +
                                  (data.message ? ` ${data.message}` : ''), 'processing');
                    }
                    const result = await (await fetch(`/result/${start.task_id}`)).json();
                    if (result.error) throw new Error(result.error);

                    this.analysisMarkdown = result.markdown;
                    this.analysisKind = kind;
                    document.getElementById('analysisTitle').textContent =
                        kind === 'minutes' ? 'Protokoll' : (name ? `Aufgaben für ${name}` : 'Aufgaben');
                    document.getElementById('analysisOutput').textContent = result.markdown;
                    document.getElementById('analysisResult').style.display = 'block';
                    setStatus('Auswertung abgeschlossen!', 'success');
                } catch (error) {
                    setStatus(`Fehler: ${error.message}`, 'error');
                } finally {
                    buttons.forEach(b => b.disabled = false);
                }
            }

            namedTranscript() {
                // Replace "Sprecher N:" line prefixes with the names entered by the user
                const names = {};
                document.querySelectorAll('#speakerNames input').forEach(input => {
                    if (input.value.trim()) names[input.dataset.speaker] = input.value.trim();
                });
                const NL = String.fromCharCode(10);
                const text = this.transcriptionResult.text.split(NL).map(line => {
                    const colon = line.indexOf(':');
                    const label = colon > 0 ? line.slice(0, colon) : '';
                    return names[label] ? names[label] + line.slice(colon) : line;
                }).join(NL);
                return text;
            }

            downloadAnalysis() {
                if (!this.analysisMarkdown) return;
                const blob = new Blob([this.analysisMarkdown], {type: 'text/markdown'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = this.analysisKind === 'minutes' ? 'protokoll.md' : 'aufgaben.md';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }

            formatTime(seconds) {
                const mins = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
            }

            showStatus(message, type) {
                const status = document.getElementById('status');
                status.textContent = message;
                status.className = `status ${type}`;
                status.style.display = 'block';
            }

            showProgress() {
                document.getElementById('progressBar').style.display = 'block';
            }

            hideProgress() {
                document.getElementById('progressBar').style.display = 'none';
            }

            downloadResult() {
                if (!this.transcriptionResult) return;

                const dataStr = JSON.stringify(this.transcriptionResult, null, 4);
                const dataBlob = new Blob([dataStr], { type: 'application/json' });
                const url = URL.createObjectURL(dataBlob);

                const link = document.createElement('a');
                link.href = url;
                link.download = `transkript_${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
            }
        }

        new WhisperApp();
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("Starting Whisper Transcription Server...")
    print("Whisper model is loaded per request and released afterwards.")
    print("Server starting on http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
