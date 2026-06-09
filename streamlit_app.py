import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st
import gymnasium as gym
import imageio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- CONFIG PAGE ---
st.set_page_config(
    page_title="DDPG Pendulum Lab - Inference",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 DDPG Inverted Pendulum Visualizer")
st.markdown("Aplikasi ringkas untuk menguji model agen DDPG yang telah dilatih pada *environment* Pendulum-v1.")

# --- PERSYARATAN FILE BOBOT ---
WEIGHTS_FILE = "pendulum_actor.weights.h5"

# --- 1. DEFINISI ARSITEKTUR AKTOR JAUH LEBIH RINGKAS ---
def get_actor_model():
    # Mengikuti dimensi state (3) dan action (1) dari Pendulum-v1
    num_states = 3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    # Output berupa aksi kontinu, skala Pendulum-v1 adalah -2.0 sampai 2.0
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    outputs = outputs * 2.0 
    
    return keras.Model(inputs, outputs)

# --- 2. FUNGSI LOAD AGEN (MEMAKAI CACHE AGAR CEPAT) ---
@st.cache_resource
def load_trained_agent(weights_path):
    if not os.path.exists(weights_path):
        return None
    try:
        model = get_actor_model()
        model.load_weights(weights_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat bobot model: {e}")
        return None

# --- 3. LOGIKA JALANKAN DEMO & AMBIL FRAMES ---
def run_agent_demo(model, max_steps=200):
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    state, _ = env.reset()
    
    frames = []
    actions = []
    total_reward = 0
    
    for _ in range(max_steps):
        # Ambil frame untuk animasi GIF nanti
        frame = env.render()
        frames.append(frame)
        
        # Prediksi aksi berdasarkan model aktor
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = model(tf_state)
        action_val = tf.squeeze(action).numpy()
        
        # Terapkan aksi ke environment
        state, reward, terminated, truncated, _ = env.step([action_val])
        total_reward += reward
        actions.append(float(action_val))
        
        if terminated or truncated:
            break
            
    env.close()
    return frames, total_reward, actions

# --- 4. ENGINE PEMBUAT GIF DENGAN IMAGEIO ---
def make_gif(frames, fps=30):
    try:
        with imageio.get_writer("live_simulation.gif", mode="I", fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        return "live_simulation.gif"
    except Exception as e:
        st.error(f"Gagal merender GIF: {e}")
        return None

# --- MAIN UI FLOW ---
if not os.path.exists(WEIGHTS_FILE):
    st.error(f"⚠️ Berkas bobot `{WEIGHTS_FILE}` tidak ditemukan di repositori! Pastikan kamu sudah mengunggah berkas .h5 hasil training.")
else:
    actor_agent = load_trained_agent(WEIGHTS_FILE)
    
    if actor_agent:
        st.success("✅ Model Aktor berhasil dimuat dari berkas bobot!")
        
        # Kontrol Interaktif Ringkas
        st.subheader("⚙️ Konfigurasi Simulasi")
        demo_steps = st.slider("Jumlah Maksimal Steps Simulasi", min_value=50, max_value=300, value=200, step=50)
        
        if st.button("🚀 Jalankan Simulasi Pendulum", use_container_width=True):
            with st.spinner("Agen sedang beraksi mengendalikan pendulum..."):
                frames, total_reward, actions = run_agent_demo(actor_agent, max_steps=demo_steps)
                gif_path = make_gif(frames)
                
            # Layout Output Hasil
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric(label="Total Reward", value=f"{total_reward:.2f}")
                st.metric(label="Rata-rata Nilai Aksi (Torque)", value=f"{np.mean(actions):.3f}")
                
                # Expandable Dataframe & Chart
                with st.expander("📊 Lihat Detail Log Aksi"):
                    df_actions = pd.DataFrame({"Step": np.arange(1, len(actions) + 1), "Aksi": actions})
                    st.line_chart(df_actions.set_index("Step"))
                    st.dataframe(df_actions, use_container_width=True, hide_index=True)
            
            with col2:
                if gif_path:
                    st.image(gif_path, caption="Animasi Hasil Kendali DDPG Agen", use_container_width=True)
                else:
                    st.warning("Gagal memuat animasi visual.")