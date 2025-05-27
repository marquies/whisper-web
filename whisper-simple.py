import torch
import os
import tempfile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import argparse
import json
import subprocess
import uuid

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Speech Recognition using Whisper Model")
parser.add_argument("video_file", type=str, help="Path to the video file")
parser.add_argument("--simple", action='store_true', help="Get whole text without timestamps")
args = parser.parse_args()

video_file = args.video_file
audio_file = tempfile.gettempdir() + "/" + str(uuid.uuid4()) + ".mp3"



# Define the ffmpeg command
#command = ["ffmpeg", "-i", video_file, "-vn", "-acodec", "copy", "output-video.aac"]
#command = ["ffmpeg", "-i", video_file,  "-q:a", "0", "-map", "a", audio_file ]
command = ["ffmpeg", "-i", video_file,"-map", "0:a", audio_file ]
# Execute the command
result = subprocess.run(command, capture_output=True, text=True)
# Capture the output and errors
output = result.stdout
error = result.stderr

output, error

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

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
    batch_size=16,
    #return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Load the video file from the command-line argument

try:
    #result = pipe(video_file)
    if args.simple:
        result = pipe(audio_file)
    else:
        result = pipe(audio_file, return_timestamps=True)
except ValueError:
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps= not args.simple,
        torch_dtype=torch_dtype,
        device=device,
    )
    #result = pipe(audio_file,return_timestamps=True,generate_kwargs = {"language":"<|en|>","task":"transcribe"})
    result = pipe(audio_file,generate_kwargs = {"language":"<|en|>","task":"transcribe"})


#print(result["chunks"])
#print(result["text"])

os.remove(audio_file)

result_json = json.dumps(result, ensure_ascii=False, indent=4)
print(result_json)
