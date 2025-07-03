import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import requests
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

import os

# Spotify API credentials (read from environment variables now)
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

# Mood labels
mood_labels = ["romantic", "city", "soft", "fun", "nostalgic", "calm", "dreamy", "sunset", "cottagecore"]

# Spotify API credentials
client_id = "bda350f22b3d4c5096d9833a256fb1d0"
client_secret = "c7d90a6d33ae4253a4a90169eed624bd"

# Function to get Spotify access token
def get_spotify_token(client_id, client_secret):
    auth_str = f"{client_id}:{client_secret}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()

    headers = {
        "Authorization": f"Basic {b64_auth_str}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}

    response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
    return response.json()["access_token"]

# Function to get song suggestions
def get_spotify_songs(query, token, limit=4):
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "track", "limit": limit}
    response = requests.get("https://api.spotify.com/v1/search", headers=headers, params=params)
    return response.json().get("tracks", {}).get("items", [])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # Process image
            image = Image.open(image_path).convert("RGB")
            inputs = clip_processor(text=mood_labels, images=image, return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            predicted_index = probs.argmax().item()
            predicted_mood = mood_labels[predicted_index]

            # Get Spotify token & songs
            spotify_token = get_spotify_token(client_id, client_secret)
            songs = get_spotify_songs(predicted_mood, spotify_token)

            # Top and more songs
            top_song = ""
            more_songs = []

            if songs:
                top_song = f"{songs[0]['name']} – {songs[0]['artists'][0]['name']}"
                for song in songs[1:4]:
                    more_songs.append(f"{song['name']} – {song['artists'][0]['name']}")

            return render_template("index.html",
                                   image_url=url_for('static', filename=f"uploads/{file.filename}"),
                                   caption=f"This image gives a {predicted_mood} vibe.",
                                   top_song=top_song,
                                   more_songs=more_songs)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
    
app.run(host="0.0.0.0", port=7860)
