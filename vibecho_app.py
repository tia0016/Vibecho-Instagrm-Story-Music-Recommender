import os
import base64
import requests
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import streamlit as st

# Load model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Mood labels
mood_labels = ["romantic", "city", "soft", "fun", "nostalgic", "calm", "dreamy", "sunset", "cottagecore"]

# Spotify credentials via environment variables
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

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

def get_spotify_songs(query, token, limit=4):
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "track", "limit": limit}
    response = requests.get("https://api.spotify.com/v1/search", headers=headers, params=params)
    return response.json().get("tracks", {}).get("items", [])

# ----------------- Streamlit UI -------------------

st.set_page_config(page_title="Vibecho 🎵", layout="centered")
st.title("🎧 Vibecho: Instagram Story Music Recommender")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    image = Image.open(uploaded_file).convert("RGB")
    inputs = clip_processor(text=mood_labels, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    predicted_index = probs.argmax().item()
    predicted_mood = mood_labels[predicted_index]

    st.success(f"🖼️ Mood detected: **{predicted_mood}**")

    # Get Spotify songs
    if client_id and client_secret:
        token = get_spotify_token(client_id, client_secret)
        songs = get_spotify_songs(predicted_mood, token)

        if songs:
            st.markdown("### 🎵 Top Recommended Songs")
            st.write(f"**Top Song:** {songs[0]['name']} – {songs[0]['artists'][0]['name']}")
            st.markdown("**More Suggestions:**")
            for song in songs[1:]:
                st.write(f"- {song['name']} – {song['artists'][0]['name']}")
        else:
            st.warning("No songs found for this mood.")
    else:
        st.error("Spotify credentials not found. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in environment.")

