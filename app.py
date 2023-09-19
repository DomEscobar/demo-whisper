# Import necessary libraries
from potassium import Potassium, Request, Response
import torch
import os
from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torchaudio

# Create a new Potassium app
app = Potassium("my_app")

# Initialize the app
@app.init
def init():

    config = WhisperConfig.from_pretrained("openai/whisper-base")
    processor = AutoProcessor.from_pretrained("openai/whisper-base")
    
    with init_empty_weights():
        model = WhisperForConditionalGeneration(config)
    model.tie_weights()

    model = load_checkpoint_and_dispatch(
        model, "model.safetensors", device_map="auto"
    )

    context = {
        "model": model,
        "processor": processor,
    }

    return context

# Handler function that accepts a file directly
@app.handler()
def handler(context: dict, request: Request) -> Response:
    device = get_device()

    # Get audio file from the request
    audio_file = request.files.get("audio")

    if not audio_file:
        return Response(
            json={"error": "Audio file not provided in the request"},
            status=400
        )

    # Load and process the audio file
    input_features = processor(load_audio(audio_file), sampling_rate=16000, return_tensors="pt").input_features.to(device)

    # Run inference on the sample
    model = context.get("model")
    generated_ids = model.generate(inputs=input_features)
    
    # Convert the generated ids back to text
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Return output JSON to the client
    return Response(
        json={"outputs": transcription}, 
        status=200
    )

# Load audio from file
def load_audio(audio_file):
    """Loads audio file into tensor and resamples to 16kHz"""
    speech, sr = torchaudio.load(audio_file)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech = resampler(speech)
    
    return speech.squeeze()

# Get the device to run inference
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Running on MPS")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    return device

if __name__ == "__main__":
    app.serve()
