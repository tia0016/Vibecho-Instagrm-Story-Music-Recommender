import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy

# ====== CONFIG ======
SPOTIFY_CLIENT_ID = "20399cb14d53499ba9da18a835e5caad"
SPOTIFY_CLIENT_SECRET = "08bdecdb688b42639a6882ce42290189"

# ====== LOAD MODELS ======
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

# ====== FUNCTIONS ======
def get_image_description(image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

def get_music_recommendations(query, limit=5):
    results = sp.search(q=query, limit=limit, type="track")
    songs = []
    for track in results['tracks']['items']:
        song_info = f"[{track['name']} - {track['artists'][0]['name']}]({track['external_urls']['spotify']})"
        songs.append(song_info)
    return songs

# ====== STREAMLIT UI ======
st.title("🎵 Image → Caption → Music 🎵")
st.write("Upload an image, see what it describes, and get Spotify music recommendations based on that description!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔎 Analyzing image..."):
        description = get_image_description(image)

    st.subheader("🖼 Image Description")
    st.write(description)

    with st.spinner("🎵 Fetching music recommendations..."):
        songs = get_music_recommendations(description)

    st.subheader("🎶 Recommended Songs")
    for idx, song in enumerate(songs, 1):
        st.markdown(f"{idx}. {song}")
