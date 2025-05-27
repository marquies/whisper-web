# app.py - Main Flask Application
from flask import Flask, request, jsonify, render_template_string
import torch
import os
import tempfile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json
import subprocess
import uuid
from werkzeug.utils import secure_filename
import threading
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Global variables for model (load once)
model = None
processor = None
pipe = None

# Progress tracking
progress_data = {}

def load_whisper_model():
    """Load Whisper model once at startup"""
    global model, processor, pipe

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"

    print("Loading Whisper model...")
    
    # Clear GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=8,  # Reduced from 16 to save memory
        torch_dtype=torch_dtype,
        device=device,
    )
    
    # Clear cache again after loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Model loaded successfully!")

def extract_audio(video_file, audio_file, task_id):
    """Extract audio from video file using FFmpeg"""
    progress_data[task_id] = {"stage": "extracting", "progress": 10}

    command = ["ffmpeg", "-i", video_file, "-map", "0:a", audio_file, "-y"]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"FFmpeg error: {result.stderr}")

    progress_data[task_id] = {"stage": "extracted", "progress": 30}

def transcribe_audio(audio_file, simple_mode, task_id):
    """Transcribe audio file using Whisper"""
    global pipe, model, processor

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
        
        # Set generation parameters
        generate_kwargs = {
            "language": "<|en|>", 
            "task": "transcribe"
            # Let the pipeline handle attention_mask automatically
        }
        
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

def process_file_async(file_path, simple_mode, task_id):
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
        result = transcribe_audio(audio_file, simple_mode, task_id)

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
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"Unexpected error: {str(e)}"
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

    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_{filename}")
    file.save(file_path)

    # Generate task ID
    task_id = str(uuid.uuid4())
    progress_data[task_id] = {"stage": "uploaded", "progress": 5}

    # Start processing in background
    thread = threading.Thread(target=process_file_async, args=(file_path, simple_mode, task_id))
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

        .download-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 5px 15px rgba(23, 162, 184, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Whisper Transkription</h1>

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
                processButton.addEventListener('click', () => this.processFile());
                downloadButton.addEventListener('click', () => this.downloadResult());
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

                if (this.transcriptionResult.chunks) {
                    container.innerHTML = this.transcriptionResult.chunks.map(chunk => `
                        <div class="timestamp-chunk">
                            <div class="timestamp">
                                ${this.formatTime(chunk.timestamp[0])} - ${this.formatTime(chunk.timestamp[1])}
                            </div>
                            <div class="text-content">${chunk.text}</div>
                        </div>
                    `).join('');
                } else {
                    container.innerHTML = `
                        <div class="simple-transcript">
                            ${this.transcriptionResult.text}
                        </div>
                    `;
                }

                results.style.display = 'block';
                downloadButton.style.display = 'inline-block';
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
    print("Loading model (this may take a few minutes)...")
    load_whisper_model()
    print("Server starting on http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)