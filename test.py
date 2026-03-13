import requests

# Test human voice
with open('human.wav', 'rb') as f:
    response = requests.post('http://localhost:5000/debug', files={'audio': f})
    print("HUMAN VOICE:")
    print(response.json())

# Test AI voice
with open('ai_voice.mp3', 'rb') as f:
    response = requests.post('http://localhost:5000/debug', files={'audio': f})
    print("\nAI VOICE:")
    print(response.json())