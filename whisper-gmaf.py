import torch
import os
import tempfile
import whisper

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, WhisperForConditionalGeneration, WhisperTokenizer
from datasets import load_dataset
import argparse
import json
import subprocess
import uuid
from typing import Optional, Collection, List, Dict
import torchaudio


def detect_language(model: WhisperForConditionalGeneration, tokenizer: WhisperTokenizer, input_features,
                    possible_languages: Optional[Collection[str]] = None) -> List[Dict[str, float]]:
    # hacky, but all language tokens and only language tokens are 6 characters long
    language_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]
    if possible_languages is not None:
        language_tokens = [t for t in language_tokens if t[2:-2] in possible_languages]
        if len(language_tokens) < len(possible_languages):
            raise RuntimeError(f'Some languages in {possible_languages} did not have associated language tokens')

    language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)

    # 50258 is the token for transcribing
    logits = model(input_features,
                   decoder_input_ids = torch.tensor([[50258] for _ in range(input_features.shape[0])])).logits
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[language_token_ids] = False
    logits[:, :, mask] = -float('inf')

    output_probs = logits.softmax(dim=-1).cpu()
    return [
        {
            lang: output_probs[input_idx, 0, token_id].item()
            for token_id, lang in zip(language_token_ids, language_tokens)
        }
        for input_idx in range(logits.shape[0])
    ]



# Parse command-line arguments
parser = argparse.ArgumentParser(description="Speech Recognition using Whisper Model")
parser.add_argument("video_file", type=str, help="Path to the video file")
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
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")

processor = AutoProcessor.from_pretrained(model_id)
audio = whisper.load_audio(audio_file)

waveform, sample_rate = torchaudio.load(str(audio_file))
arr = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)

input_features = processor(arr.squeeze().numpy(), sampling_rate=16000,
                               return_tensors="pt").input_features

#input_features = processor(audio, return_tensors="pt", sampling_rate=16000).input_features

#language_name = detect_language_tokens(model, tokenizer, input_features, {'en', 'de'})
#print(detect_language(model, tokenizer, input_features, {'en', 'de'}))

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
    result = pipe(audio_file, return_timestamps=True, generate_kwargs = {"no_speech_threshold":0.6, "logprob_threshold": -1.0})
except ValueError:
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
    result = pipe(audio_file,return_timestamps=True,generate_kwargs = {"language":"<|en|>","task":"transcribe", "logprob_threshold": -1.0})


#print(result["chunks"])
#print(result["text"])
from langdetect import detect
detected_language = detect(result['text'])
result['language'] = detected_language
os.remove(audio_file)

result_json = json.dumps(result, ensure_ascii=False, indent=4)
print(result_json)
