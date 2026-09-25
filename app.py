import os
import json
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import streamlit as st

from src.model import WhaleSEDModel
from src.dataset import CLASS_MAPPING, get_device
from src.infer import run_inference_on_file, write_raven_selections

# Page configuration
st.set_page_config(
    page_title="Baleen Whale Sound Event Detection",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styles (font imports, glassmorphism, entry animations, glow shadows)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        /* Global style configurations */
        html, body, [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top right, #111827, #030712) !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            color: #f3f4f6;
        }

        /* Entry Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(15px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes indicatorPulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }

        .fadeIn {
            animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        /* Header Layout */
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 24px;
            margin-bottom: 30px;
        }

        .system-status {
            display: flex;
            align-items: center;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            padding: 6px 14px;
            border-radius: 24px;
            font-size: 13px;
            font-weight: 500;
            color: #34d399;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            margin-right: 8px;
            animation: indicatorPulse 2s infinite ease-in-out;
        }

        /* Glassmorphism Metric Cards */
        .metric-card {
            background: rgba(31, 41, 55, 0.35) !important;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            border-color: rgba(0, 242, 254, 0.3);
            box-shadow: 0 12px 40px rgba(0, 242, 254, 0.12);
        }

        .metric-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 36px;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #a5f3fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        }

        .metric-label {
            font-size: 12px;
            color: #9ca3af;
            font-weight: 600;
            letter-spacing: 0.8px;
            text-transform: uppercase;
        }

        /* Streamlit Input Styling Override */
        div[data-testid="stSidebar"] {
            background-color: #030712 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
            color: #030712 !important;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 24px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 242, 254, 0.2);
        }

        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 25px rgba(0, 242, 254, 0.45);
        }
    </style>
""", unsafe_allow_html=True)

# Helper to load model and thresholds
@st.cache_resource
def load_detection_model(model_path):
    device = get_device()
    model = WhaleSEDModel(num_classes=len(CLASS_MAPPING))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.to(device)
        model.eval()
        
        # Load thresholds
        thresholds_path = model_path.replace(".pth", "_thresholds.json")
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r', encoding='utf-8') as f:
                thresholds = json.load(f)
        else:
            thresholds = {cls: 0.5 for cls in CLASS_MAPPING.keys()}
        return model, thresholds, device
    return None, None, device

# Visualizing function
def plot_spectrogram_with_events(y, sr, events, offset_sec=0.0):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    
    # Compute STFT magnitude
    stft = np.abs(librosa.stft(y, n_fft=256, hop_length=64))
    spec_db = librosa.amplitude_to_db(stft, ref=np.max)
    
    # Draw spectrogram
    img = librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=64,
        x_axis='time',
        y_axis='linear',
        ax=ax,
        cmap='magma',
        vmax=0,
        vmin=-80
    )
    
    # Focus on baleen whale call range (10 - 100 Hz)
    ax.set_ylim(10, 100)
    ax.set_title("Low-Frequency Spectrogram & Predicted Detections (10 - 100 Hz)", color="#00f2fe", fontsize=12, pad=10)
    ax.set_xlabel("Time (seconds)", color="#9ca3af")
    ax.set_ylabel("Frequency (Hz)", color="#9ca3af")
    
    # Stylize axes
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#374151')
    ax.spines['left'].set_color('#374151')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    
    # Draw boxes for active events
    colors = {
        'Bm.Ant-A': '#00f2fe',
        'Bm.Ant-B': '#4facfe',
        'Bm.Ant-Z': '#00ff87',
        'Bm.D': '#ff007f',
        'Bp.20Hz': '#ff0844',
        'Bp.20Plus': '#ffb199',
        'Bp.Downsweep': '#f12711',
        'Unidentified': '#f5af19'
    }
    
    for ev in events:
        # Get start/end relative to snippet window
        start_rel = ev['begin_time_sec'] - offset_sec
        end_rel = ev['end_time_sec'] - offset_sec
        
        # Only draw if in range
        if start_rel >= 0 or end_rel <= len(y)/sr:
            cls = ev['class']
            color = colors.get(cls, '#00f2fe')
            
            rect = plt.Rectangle(
                (start_rel, 15),
                end_rel - start_rel,
                80,
                fill=False,
                edgecolor=color,
                linewidth=2.0,
                linestyle='-'
            )
            ax.add_patch(rect)
            ax.text(
                start_rel,
                90,
                f"{cls} ({ev['confidence']:.2f})",
                color=color,
                fontsize=9,
                fontweight='bold',
                bbox=dict(facecolor='#111827', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.2')
            )
            
    plt.tight_layout()
    return fig

def main():
    # Glowing Title Container
    st.markdown("""
        <div class="header-container fadeIn">
            <div>
                <h1 style="margin:0; font-family:'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🐋 Baleen Whale SED Platform</h1>
                <p style="margin:5px 0 0 0; color:#9ca3af; font-size:1rem; font-weight: 400;">Automated Real-Time Localization and Classification of Southern Ocean Acoustics</p>
            </div>
            <div class="system-status">
                <span class="status-dot"></span>
                <span>System Active (MPS)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_path = "models/best_model.pth"
    model, thresholds, device = load_detection_model(model_path)
    
    if model is None:
        st.error("No model checkpoint found at models/best_model.pth. Please run the training pipeline first.")
        return
        
    # Sidebar control panel
    st.sidebar.header("🛠️ Pipeline Controls")
    
    # Audio selection options
    audio_source = st.sidebar.radio("Select Audio Source", ["Preloaded Sample", "Upload WAV File"])
    
    # Preloaded files list
    preloaded_dir = "data/Greenwich64S2015/wav"
    preloaded_files = []
    if os.path.exists(preloaded_dir):
        preloaded_files = sorted(glob.glob(os.path.join(preloaded_dir, "*.wav")))
        
    selected_wav_path = None
    
    if audio_source == "Preloaded Sample":
        if preloaded_files:
            file_names = [os.path.basename(f) for f in preloaded_files]
            selected_file = st.sidebar.selectbox("Choose a WAV File", file_names)
            selected_wav_path = os.path.join(preloaded_dir, selected_file)
        else:
            st.sidebar.warning("No preloaded WAV files found in data/Greenwich64S2015/wav.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a hydrophone WAV recording", type=["wav"])
        if uploaded_file is not None:
            # Save uploaded file temporarily
            os.makedirs("scratch", exist_ok=True)
            selected_wav_path = os.path.join("scratch", uploaded_file.name)
            with open(selected_wav_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
    if selected_wav_path:
        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Confidence Thresholds")
        
        # Allow adjusting optimal thresholds dynamically
        adjusted_thresholds = {}
        for cls_name, opt_th in thresholds.items():
            adjusted_thresholds[cls_name] = st.sidebar.slider(
                f"{cls_name} threshold",
                min_value=0.01,
                max_value=0.99,
                value=opt_th,
                step=0.01
            )
            
        # Analysis trigger
        run_btn = st.sidebar.button("⚡ Run Sound Event Detection", use_container_width=True)
        
        if run_btn:
            with st.spinner("Processing audio file, computing spectrogram and running inference..."):
                # Run sliding window inference
                events = run_inference_on_file(
                    model=model,
                    wav_path=selected_wav_path,
                    device=device,
                    thresholds=adjusted_thresholds
                )
                
                # Save events in session state
                st.session_state["events"] = events
                st.session_state["wav_path"] = selected_wav_path
                
        # Main results pane
        if "events" in st.session_state and st.session_state["wav_path"] == selected_wav_path:
            events = st.session_state["events"]
            
            # Row 1: Premium Glassmorphism Summary Cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                    <div class="metric-card fadeIn">
                        <div class="metric-value">{len(events)}</div>
                        <div class="metric-label">Total Detections</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                # Find unique species
                unique_spp = len(set(ev['class'] for ev in events))
                st.markdown(f"""
                    <div class="metric-card fadeIn" style="border-top: 3px solid #00ff87;">
                        <div class="metric-value">{unique_spp}</div>
                        <div class="metric-label">Whale Categories</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                # Avg confidence
                avg_conf = np.mean([ev['confidence'] for ev in events]) if events else 0.0
                st.markdown(f"""
                    <div class="metric-card fadeIn" style="border-top: 3px solid #ff007f;">
                        <div class="metric-value">{avg_conf:.2f}</div>
                        <div class="metric-label">Mean Confidence</div>
                    </div>
                """, unsafe_allow_html=True)
            with col4:
                # File duration
                with sf.SoundFile(selected_wav_path) as sf_file:
                    dur_min = (len(sf_file) / sf_file.samplerate) / 60.0
                st.markdown(f"""
                    <div class="metric-card fadeIn" style="border-top: 3px solid #f5af19;">
                        <div class="metric-value">{dur_min:.1f} min</div>
                        <div class="metric-label">Duration</div>
                    </div>
                """, unsafe_allow_html=True)
                
            # Row 2: Spectrogram Visualizer
            st.markdown("<h3 class='fadeIn'>🔍 Interactive Spectrogram Inspector</h3>", unsafe_allow_html=True)
            
            # Duration slider to select snippet to zoom in
            tot_dur_sec = dur_min * 60.0
            zoom_offset = st.slider("Select segment start time (seconds)", 0, int(tot_dur_sec - 30), 0)
            
            # Read and plot snippet
            with sf.SoundFile(selected_wav_path) as sf_file:
                sr = sf_file.samplerate
                sf_file.seek(int(zoom_offset * sr))
                y_snippet = sf_file.read(int(30.0 * sr))
                
            if len(y_snippet.shape) > 1:
                y_snippet = y_snippet[:, 0]
                
            # Filter and plot
            stft_fig = plot_spectrogram_with_events(y_snippet, sr, events, offset_sec=zoom_offset)
            st.pyplot(stft_fig)
            
            # Playback snippet
            st.markdown("<h4 class='fadeIn'>🎧 Listen to Audio Snippet</h4>", unsafe_allow_html=True)
            col_play1, col_play2 = st.columns([2, 1])
            with col_play1:
                # Standard playback
                st.audio(y_snippet, sample_rate=sr)
            with col_play2:
                # Pitch shift slider
                st.info("💡 Baleen calls are very low frequency. If they are hard to hear, you can speed up standard playback in the audio player controls.")
                
            # Row 3: Selections Table & Export
            st.markdown("<h3 class='fadeIn'>📋 Detected Calls List</h3>", unsafe_allow_html=True)
            if events:
                df = pd.DataFrame(events)
                df = df.rename(columns={
                    'class': 'Species/Call Type',
                    'begin_time_sec': 'Start Time (s)',
                    'end_time_sec': 'End Time (s)',
                    'begin_samp': 'Start Sample',
                    'end_samp': 'End Sample',
                    'confidence': 'Confidence Score'
                })
                st.dataframe(df[['Species/Call Type', 'Start Time (s)', 'End Time (s)', 'Confidence Score']], use_container_width=True)
                
                # Export to Raven Selections
                output_name = os.path.basename(selected_wav_path).replace(".wav", ".predictions.selections.txt")
                scratch_dir = "scratch"
                os.makedirs(scratch_dir, exist_ok=True)
                out_path = os.path.join(scratch_dir, output_name)
                
                write_raven_selections(events, out_path, os.path.basename(selected_wav_path))
                
                with open(out_path, "r") as f:
                    selections_data = f.read()
                    
                st.download_button(
                    label="📥 Export to Raven Selections Table",
                    data=selections_data,
                    file_name=output_name,
                    mime="text/plain"
                )
            else:
                st.info("No calls detected above the selected thresholds for this recording.")
                
        else:
            # Initial placeholder
            st.info("👈 Configure options in the sidebar and click 'Run Sound Event Detection' to analyze this hydrophone recording.")

if __name__ == "__main__":
    main()
