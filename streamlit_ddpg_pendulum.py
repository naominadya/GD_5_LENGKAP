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

st.set_page_config(
    page_title="DDPG Pendulum Lab",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 DDPG Inverted Pendulum Visualizer")
st.markdown("Aplikasi ringkas untuk menguji performa model agen DDPG pada *environment* Pendulum-v1.")

# Nama berkas dipastikan sama dengan yang sudah di-rename
WEIGHTS_FILE = "pendulum_actor.weights.h5"

def get_actor_model():
    num_states = 3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    outputs = outputs * 2.0 
    
    return keras.Model(inputs, outputs)

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

def run_agent_demo(model, max_steps=200):
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    state, _ = env.reset()
    
    frames = []
    actions = []
    total_reward = 0
    
    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)
        
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = model(tf_state)
        action_val = tf.squeeze(action).numpy()
        
        state, reward, terminated, truncated, _ = env.step([action_val])
        total_reward += reward
        actions.append(float(action_val))
        
        if terminated or truncated:
            break
            
    env.close()
    return frames, total_reward, actions

if not os.path.exists(WEIGHTS_FILE):
    st.error(f"⚠️ Berkas bobot `{WEIGHTS_FILE}` tidak ditemukan di repositori! Pastikan berkas .h5 sudah diunggah.")
else:
    actor_agent = load_trained_agent(WEIGHTS_FILE)
    
    if actor_agent:
        st.success("✅ Model Aktor berhasil dimuat dari berkas repositori!")
        
        st.subheader("⚙️ Konfigurasi Simulasi")
        demo_steps = st.slider("Jumlah Maksimal Steps Simulasi", min_value=50, max_value=300, value=200, step=50)
        
        if st.button("🚀 Jalankan Simulasi Pendulum", use_container_width=True):
            with st.spinner("Agen sedang beraksi mengendalikan pendulum..."):
                frames, total_reward, actions = run_agent_demo(actor_agent, max_steps=demo_steps)
                gif_path = "live_simulation.gif"
                imageio.mimsave(gif_path, frames, fps=30)
                
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric(label="Total Reward", value=f"{total_reward:.2f}")
                st.metric(label="Rata-rata Nilai Aksi (Torque)", value=f"{np.mean(actions):.3f}")
                with st.expander("📊 Lihat Detail Log Aksi"):
                    st.line_chart(actions)
            
            with col2:
                st.image(gif_path, caption="Animasi Hasil Kendali DDPG Agen", use_container_width=True)