import streamlit as st
import torch
from demucs.pretrained import get_model
from demucs.audio import AudioFile, save_audio
import numpy as np
import os
import tempfile
from pathlib import Path
import torchaudio
import warnings

# Filter out specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')

class MusicSeparator:
    def __init__(self):
        self.model = get_model('htdemucs')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
    def separate_track(self, audio_path):
        # Load the audio file
        wav = AudioFile(audio_path).read(streams=0, samplerate=44100, channels=2)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        
        # Convert to tensor properly
        wav = torch.from_numpy(wav).float()
        
        # Separate the audio
        with torch.no_grad():
            wav = wav.to(self.device)
            sources = self.model.forward(wav[None])
            sources = sources.cpu()
        
        # Get the separated parts
        sources = sources[0].numpy()
        drums, bass, other, vocals = sources
        
        # Combine drums, bass, and other for instrumental
        instrumental = drums + bass + other
        
        return vocals, instrumental

def save_audio_file(audio_data, sample_rate, output_path):
    # Convert to tensor if numpy array
    if isinstance(audio_data, np.ndarray):
        audio_tensor = torch.from_numpy(audio_data)
    else:
        audio_tensor = audio_data
        
    # Ensure audio tensor is float32
    audio_tensor = audio_tensor.float()
    
    # Save using torchaudio with backend specified
    torchaudio.save(
        output_path,
        audio_tensor,
        sample_rate,
        backend="soundfile"
    )

def main():
    st.set_page_config(
        page_title="Music Source Separator",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Music Source Separation App")
    st.write("Upload a song to separate vocals and instrumental parts")
    
    # Add some CSS to improve the appearance
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            margin-top: 20px;
        }
        .stProgress>div>div>div {
            background-color: #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav'])
    
    # Initialize session state
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    if uploaded_file is not None:
        # Create a temporary directory to store the uploaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file to temporary directory
            temp_path = Path(temp_dir) / "input_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Initialize progress tracking
            progress_text = "Operation in progress. Please wait..."
            my_bar = st.progress(0, text=progress_text)
            
            try:
                # Initialize separator
                my_bar.progress(30, text="Loading model...")
                separator = MusicSeparator()
                
                # Separate tracks
                my_bar.progress(50, text="Separating audio tracks...")
                vocals, instrumental = separator.separate_track(str(temp_path))
                
                # Save separated tracks
                my_bar.progress(80, text="Saving separated tracks...")
                vocals_path = Path(temp_dir) / "vocals.wav"
                instrumental_path = Path(temp_dir) / "instrumental.wav"
                
                save_audio_file(vocals, 44100, str(vocals_path))
                save_audio_file(instrumental, 44100, str(instrumental_path))
                
                my_bar.progress(100, text="Separation completed! 🎉")
                st.session_state.processing_complete = True
                
                # Create columns for the audio players
                col1, col2 = st.columns(2)
                
                # Display audio players with download buttons
                with col1:
                    st.subheader("🎤 Vocals")
                    with open(vocals_path, 'rb') as f:
                        vocals_bytes = f.read()
                        st.audio(vocals_bytes, format='audio/wav')
                        st.download_button(
                            label="Download Vocals",
                            data=vocals_bytes,
                            file_name="vocals.wav",
                            mime="audio/wav"
                        )
                
                with col2:
                    st.subheader("🎹 Instrumental")
                    with open(instrumental_path, 'rb') as f:
                        instrumental_bytes = f.read()
                        st.audio(instrumental_bytes, format='audio/wav')
                        st.download_button(
                            label="Download Instrumental",
                            data=instrumental_bytes,
                            file_name="instrumental.wav",
                            mime="audio/wav"
                        )
                
                # Clear progress bar after successful completion
                if st.session_state.processing_complete:
                    my_bar.empty()
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please try again with a different audio file or check if the file is corrupted.")
                my_bar.empty()

if __name__ == "__main__":
    main()