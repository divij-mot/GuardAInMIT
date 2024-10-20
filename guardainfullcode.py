
import time
from time import sleep
import base64
import requests
import PIL
from PIL import Image
import numpy as np
from openai import OpenAI
from google.cloud import texttospeech
import subprocess
from picamera2 import Picamera2 



# Initialize PiCamera2
picamera2 = Picamera2()

# Configure for a high-resolution still capture
config = picamera2.create_still_configuration()
config["main"]["size"] = (3280, 2464)  # Set resolution

# Apply the configuration
picamera2.configure(config)

# Allow the camera some time to adjust to conditions
time.sleep(1)

# Capture the image
picamera2.start()
image = picamera2.capture_array()
picamera2.stop()

# Save the image
from PIL import Image
img = Image.fromarray(image)
img.save("/home/guardain/guardAInCodebase/image.png")


def play_audio_with_ffmpeg(audio_file_path):
    """Play an audio file using ffplay."""
    subprocess.run(['ffplay', '-nodisp', '-autoexit', audio_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def record_audio_with_ffmpeg(output_file_path, duration_seconds):
    """Record audio from the default microphone for a specified duration and save it to an output file."""
    subprocess.run([
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-f', 'alsa',  # Change 'alsa' to 'avfoundation' for macOS or 'dshow' for Windows if needed
        '-i', 'default',  # Change 'default' to specific device name if needed
        '-t', str(duration_seconds),
        '-acodec', 'mp3',  # Specify audio codec
        '-ar', '44100',  # Set audio sample rate
        '-ac', '1',  # Set number of audio channels
        output_file_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)



# Paths to the beep signals and the output file
start_signal = "/home/guardain/guardAInCodebase/1tonebeep.mp3"
end_signal = "/home/guardain/guardAInCodebase/2tonebeep.mp3"
output_file = "/home/guardain/guardAInCodebase/audioclip.mp3"
record_duration = 4  # Duration to record in seconds

# Play start signal
print("Playing start signal...")
play_audio_with_ffmpeg(start_signal)

# Record audio
# print("Recording...")
time.sleep(0.7)
record_audio_with_ffmpeg(output_file, record_duration)
print("Finished recording.")

# Play end signal
print("Playing end signal...")
play_audio_with_ffmpeg(end_signal)

print(f"File {output_file} saved.")


api_key = ""
# OpenAI API Key
client = OpenAI(api_key = "")




audio_file = open("/home/guardain/guardAInCodebase/audioclip.mp3", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  temperature="0.2",
  file=audio_file, 
  response_format="text"
)
print(transcription)

new_height = 768

# Open  image
img = Image.open('/home/guardain/guardAInCodebase/image.png')

# Calculate the new width -- used for openai vision so that their server doesnt have to process the image
width, height = img.size
aspect_ratio = width / height
new_width = int(new_height * aspect_ratio)

# Resize the image 
resized_img = img.resize((new_width, new_height), Image.LANCZOS)

# Convert the image to RGB mode 
resized_img_rgb = resized_img.convert('RGB')

# Save the resized and converted image
resized_img_rgb.save('/home/guardain/guardAInCodebase/resized_image.jpg')

#  encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


# Getting base64
base64_image = encode_image('/home/guardain/guardAInCodebase/resized_image.jpg')
print("encoded")


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4-vision-preview",
  "temperature": 0.1,
  "messages": [
    {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": '''You are assisting a visually impaired user by analyzing images and answering t
            heir questions based on the image content. Your responses must be clear, accurate, 
            and MUST BE confined to 1-3 sentences, providing essential information or 
            insights directly related to the user's query and the image content.

Image Analysis: Start by describing the key elements of the image relevant to the user's query.
Answer Precision: Directly address the user's question based on the image analysis, ensuring the response is relevant and concise.
Brevity is Key: Limit your response to NO MORE THAN three sentences, focusing on delivering the most valuable information, 
and answering the user's question succinctly.
'''
            }
        ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": transcription
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 2000
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

response_json = response.json()


assistant_message_content = response_json['choices'][0]['message']['content']

print(assistant_message_content)

#Google Cloud TTS


client = texttospeech.TextToSpeechClient()


synthesis_input = texttospeech.SynthesisInput(text=assistant_message_content)

# Select the voice as neural 2
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US", name="en-US-Neural2-D"
)

# Select the type of audio file you want returned
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=1.25
)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
response = client.synthesize_speech(
    input=synthesis_input, voice=voice, audio_config=audio_config
)

# Binary audio response
with open("/home/guardain/guardAInCodebase/output.mp3", "wb") as out:
    # Write the response to the output file.
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')

# Play the audio file
play_audio_with_ffmpeg("/home/guardain/guardAInCodebase/output.mp3")


