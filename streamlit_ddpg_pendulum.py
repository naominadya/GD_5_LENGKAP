import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import io
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import gymnasium as gym
    DEPENDENCY_ERROR = None
except Exception as exc:
    tf = None
    keras = None
    layers = None
    gym = None
    DEPENDENCY_ERROR = exc


st.set_page_config(
    page_title="DDPG Pendulum Lab",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main > div { padding-top: 1.5rem; }
    .hero-box {
        padding: 1.35rem 1.5rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #eef5ff 0%, #f7f1ff 55%, #fff8ea 100%);
        border: 1px solid rgba(120, 120, 120, 0.18);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: .25rem;
        color: #222222;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #444444;
        line-height: 1.55;
        max-width: 980px;
    }
    .mini-card {
        padding: 1rem;
        border-radius: 18px;
        border: 1px solid rgba(120, 120, 120, 0.18);
        background: rgba(255,255,255,0.72);
        box-shadow: 0 4px 18px rgba(0,0,0,0.04);
    }
    .section-label {
        font-size: .82rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #6b7280;
        font-weight: 700;
        margin-bottom: .35rem;
    }
    .big-number {
        font-size: 1.65rem;
        font-weight: 800;
        color: #111827;
    }
    .muted { color: #6b7280; font-size: .92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🎯 DDPG Pendulum Lab</div>
        <div class="hero-subtitle">
            Aplikasi mini untuk melatih agen <b>Deep Deterministic Policy Gradient</b> pada environment
            <b>Pendulum-v1</b>. Di sini bisa mengatur hyperparameter, menjalankan training, melihat grafik reward,
            menyimpan bobot model, dan mencoba hasil agen secara langsung.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if DEPENDENCY_ERROR is not None:
    st.error("Library yang dibutuhkan belum lengkap atau belum berhasil dimuat.")
    st.code(
        "pip install streamlit tensorflow gymnasium[classic_control] matplotlib numpy pandas imageio",
        language="bash",
    )
    st.caption(f"Detail error: {DEPENDENCY_ERROR}")
    st.stop()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class ReplayBuffer:
    def __init__(self, num_states, num_actions, buffer_capacity=50_000, batch_size=64):
        self.buffer_capacity = int(buffer_capacity)
        self.batch_size = int(batch_size)
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, num_states), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states), dtype=np.float32)

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def sample(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        if record_range == 0:
            return None

        batch_indices = np.random.choice(record_range, self.batch_size)
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)
        return state_batch, action_batch, reward_batch, next_state_batch


class DDPGAgent:
    def __init__(
        self,
        num_states,
        num_actions,
        upper_bound,
        lower_bound,
        actor_lr=0.001,
        critic_lr=0.002,
        gamma=0.99,
        tau=0.005,
        std_dev=0.2,
        buffer_capacity=50_000,
        batch_size=64,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.upper_bound = float(upper_bound)
        self.lower_bound = float(lower_bound)
        self.gamma = float(gamma)
        self.tau = float(tau)

        self.noise = OUActionNoise(
            mean=np.zeros(num_actions),
            std_deviation=float(std_dev) * np.ones(num_actions),
        )

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()
        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.actor_optimizer = keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=critic_lr)
        self.buffer = ReplayBuffer(num_states, num_actions, buffer_capacity, batch_size)

    def get_actor(self):
        last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(
            self.num_actions,
            activation="tanh",
            kernel_initializer=last_init,
        )(out)
        outputs = layers.Lambda(lambda x: x * self.upper_bound)(outputs)
        return keras.Model(inputs, outputs, name="Actor")

    def get_critic(self):
        state_input = layers.Input(shape=(self.num_states,))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        action_input = layers.Input(shape=(self.num_actions,))
        action_out = layers.Dense(32, activation="relu")(action_input)

        concat = layers.Concatenate()([state_out, action_out])
        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)
        return keras.Model([state_input, action_input], outputs, name="Critic")

    def policy(self, state, use_noise=True):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        sampled_actions = tf.squeeze(self.actor_model(state_tensor)).numpy()
        sampled_actions = np.array(sampled_actions, ndmin=1)

        if use_noise:
            sampled_actions = sampled_actions + self.noise()

        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        return legal_action.astype(np.float32)

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            actor_loss = -tf.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )
        return actor_loss, critic_loss

    def learn(self):
        batch = self.buffer.sample()
        if batch is None:
            return None, None
        return self.update(*batch)

    def update_targets(self):
        actor_weights = self.actor_model.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        critic_weights = self.critic_model.get_weights()
        target_critic_weights = self.target_critic.get_weights()

        new_target_actor = [
            self.tau * current + (1 - self.tau) * target
            for current, target in zip(actor_weights, target_actor_weights)
        ]
        new_target_critic = [
            self.tau * current + (1 - self.tau) * target
            for current, target in zip(critic_weights, target_critic_weights)
        ]

        self.target_actor.set_weights(new_target_actor)
        self.target_critic.set_weights(new_target_critic)


def get_env_specs():
    env = gym.make("Pendulum-v1")
    specs = {
        "num_states": env.observation_space.shape[0],
        "num_actions": env.action_space.shape[0],
        "upper_bound": float(env.action_space.high[0]),
        "lower_bound": float(env.action_space.low[0]),
    }
    env.close()
    return specs


def train_agent(settings):
    set_seed(settings["seed"])
    env = gym.make("Pendulum-v1")
    specs = get_env_specs()

    agent = DDPGAgent(
        num_states=specs["num_states"],
        num_actions=specs["num_actions"],
        upper_bound=specs["upper_bound"],
        lower_bound=specs["lower_bound"],
        actor_lr=settings["actor_lr"],
        critic_lr=settings["critic_lr"],
        gamma=settings["gamma"],
        tau=settings["tau"],
        std_dev=settings["std_dev"],
        buffer_capacity=settings["buffer_capacity"],
        batch_size=settings["batch_size"],
    )

    rewards = []
    avg_rewards = []
    actor_losses = []
    critic_losses = []

    progress = st.progress(0)
    status = st.empty()
    chart_placeholder = st.empty()

    for ep in range(settings["total_episodes"]):
        prev_state, _ = env.reset(seed=settings["seed"] + ep)
        agent.noise.reset()
        episodic_reward = 0.0

        for _ in range(settings["max_steps"]):
            action = agent.policy(prev_state, use_noise=True)
            state, reward, done, truncated, _ = env.step(action)

            agent.buffer.record((prev_state, action, reward, state))
            actor_loss, critic_loss = agent.learn()
            agent.update_targets()

            if actor_loss is not None and critic_loss is not None:
                actor_losses.append(float(actor_loss.numpy()))
                critic_losses.append(float(critic_loss.numpy()))

            episodic_reward += float(reward)
            prev_state = state

            if done or truncated:
                break

        rewards.append(episodic_reward)
        avg_rewards.append(float(np.mean(rewards[-40:])))

        progress.progress((ep + 1) / settings["total_episodes"])
        status.info(
            f"Episode {ep + 1}/{settings['total_episodes']} • "
            f"Reward: {episodic_reward:.2f} • "
            f"Rata-rata 40 episode: {avg_rewards[-1]:.2f}"
        )

        if (ep + 1) % settings["chart_interval"] == 0 or ep == settings["total_episodes"] - 1:
            df_live = pd.DataFrame(
                {
                    "Episode": np.arange(1, len(avg_rewards) + 1),
                    "Rata-rata Reward": avg_rewards,
                }
            ).set_index("Episode")
            chart_placeholder.line_chart(df_live)

    env.close()

    history = pd.DataFrame(
        {
            "Episode": np.arange(1, len(rewards) + 1),
            "Reward": rewards,
            "Rata-rata Reward 40 Episode": avg_rewards,
        }
    )

    losses = pd.DataFrame(
        {
            "Step": np.arange(1, len(actor_losses) + 1),
            "Actor Loss": actor_losses,
            "Critic Loss": critic_losses,
        }
    )

    return agent, history, losses, specs


def run_agent_demo(agent, seed=123, max_steps=200):
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    state, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    actions = []

    for _ in range(max_steps):
        action = agent.policy(state, use_noise=False)
        state, reward, done, truncated, _ = env.step(action)
        total_reward += float(reward)
        actions.append(float(action[0]))

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if done or truncated:
            break

    env.close()
    return frames, total_reward, actions


def make_gif(frames, duration=0.04):
    try:
        import imageio.v2 as imageio
    except Exception:
        return None

    if not frames:
        return None

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        imageio.mimsave(tmp.name, frames[::2], duration=duration)
        return tmp.name


def make_weights_zip(agent):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        paths = {
            "pendulum_actor.weights.h5": tmp_path / "pendulum_actor.weights.h5",
            "pendulum_critic.weights.h5": tmp_path / "pendulum_critic.weights.h5",
            "pendulum_target_actor.weights.h5": tmp_path / "pendulum_target_actor.weights.h5",
            "pendulum_target_critic.weights.h5": tmp_path / "pendulum_target_critic.weights.h5",
        }

        agent.actor_model.save_weights(paths["pendulum_actor.weights.h5"])
        agent.critic_model.save_weights(paths["pendulum_critic.weights.h5"])
        agent.target_actor.save_weights(paths["pendulum_target_actor.weights.h5"])
        agent.target_critic.save_weights(paths["pendulum_target_critic.weights.h5"])

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for name, path in paths.items():
                zip_file.write(path, arcname=name)
        buffer.seek(0)
        return buffer.getvalue()


specs = get_env_specs()

with st.sidebar:
    st.header("⚙️ Pengaturan")
    st.caption("Nilai awal mengikuti notebook, tetapi dibuat fleksibel agar mudah dicoba.")

    total_episodes = st.slider("Jumlah episode", 5, 300, 100, 5)
    max_steps = st.slider("Maksimal step per episode", 50, 300, 200, 10)
    std_dev = st.slider("OU noise / eksplorasi", 0.0, 0.8, 0.2, 0.05)

    st.divider()
    actor_lr = st.number_input("Actor learning rate", min_value=0.00001, max_value=0.05, value=0.001, format="%.5f")
    critic_lr = st.number_input("Critic learning rate", min_value=0.00001, max_value=0.05, value=0.002, format="%.5f")
    gamma = st.slider("Gamma", 0.80, 0.999, 0.99, 0.001)
    tau = st.slider("Tau", 0.001, 0.05, 0.005, 0.001)

    st.divider()
    buffer_capacity = st.select_slider("Replay buffer", options=[10_000, 25_000, 50_000, 100_000], value=50_000)
    batch_size = st.select_slider("Batch size", options=[32, 64, 128], value=64)
    seed = st.number_input("Seed", value=42, step=1)
    chart_interval = st.select_slider("Update grafik setiap", options=[1, 5, 10, 20], value=5)

settings = {
    "total_episodes": int(total_episodes),
    "max_steps": int(max_steps),
    "std_dev": float(std_dev),
    "actor_lr": float(actor_lr),
    "critic_lr": float(critic_lr),
    "gamma": float(gamma),
    "tau": float(tau),
    "buffer_capacity": int(buffer_capacity),
    "batch_size": int(batch_size),
    "seed": int(seed),
    "chart_interval": int(chart_interval),
}

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="section-label">State</div>
            <div class="big-number">{specs['num_states']}</div>
            <div class="muted">cos(θ), sin(θ), θ dot</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="section-label">Aksi</div>
            <div class="big-number">{specs['num_actions']}</div>
            <div class="muted">torsi kontinu</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_c:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="section-label">Batas Aksi</div>
            <div class="big-number">{specs['lower_bound']:.0f} sampai {specs['upper_bound']:.0f}</div>
            <div class="muted">nilai dijaga dengan clipping</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_d:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="section-label">Episode</div>
            <div class="big-number">{settings['total_episodes']}</div>
            <div class="muted">sesuai pengaturan sidebar</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("🧭 Alur Kerja")
    st.markdown(
        """
        1. **Actor** memilih aksi berdasarkan state pendulum.  
        2. **Noise Ornstein-Uhlenbeck** ditambahkan supaya agen berani mencoba gerakan baru.  
        3. Pengalaman disimpan ke **Replay Buffer** dalam bentuk `(state, action, reward, next_state)`.  
        4. **Critic** menilai kualitas aksi, lalu Actor dan Critic diperbarui.  
        5. **Target Network** diperbarui perlahan memakai `tau` agar training lebih stabil.
        """
    )

with right:
    st.subheader("📌 Target Training")
    st.markdown(
        """
        Semakin baik agen mengendalikan pendulum, nilai reward akan semakin mendekati nol.
        Grafik yang naik menunjukkan performa agen mulai membaik dari episode ke episode.
        """
    )
    st.info("Pendulum-v1 memberi reward negatif. Jadi, reward yang lebih tinggi berarti hasilnya lebih baik.")

with st.expander("Lihat struktur model Actor dan Critic"):
    model_col_1, model_col_2 = st.columns(2)
    with model_col_1:
        st.markdown("**Actor**")
        st.code(
            "State(3) → Dense(256, ReLU) → Dense(256, ReLU) → Dense(1, Tanh) × batas aksi",
            language="text",
        )
    with model_col_2:
        st.markdown("**Critic**")
        st.code(
            "State + Action → Concatenate → Dense(256, ReLU) → Dense(256, ReLU) → Q-Value(1)",
            language="text",
        )

st.divider()
st.subheader("🚀 Training")

train_col, info_col = st.columns([0.62, 0.38])
with train_col:
    run_training = st.button("Mulai Training", type="primary", use_container_width=True)
with info_col:
    st.caption("Training akan berjalan langsung di aplikasi. Untuk percobaan cepat, gunakan 5–20 episode terlebih dahulu.")

if run_training:
    agent, history, losses, specs = train_agent(settings)
    st.session_state["agent"] = agent
    st.session_state["history"] = history
    st.session_state["losses"] = losses
    st.session_state["specs"] = specs
    st.success("Training selesai. Hasil sudah tersedia di bawah.")

if "history" in st.session_state:
    history = st.session_state["history"]
    losses = st.session_state["losses"]
    agent = st.session_state["agent"]

    st.subheader("📈 Hasil Reward")
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Reward episode terakhir", f"{history['Reward'].iloc[-1]:.2f}")
    metric_2.metric("Rata-rata reward akhir", f"{history['Rata-rata Reward 40 Episode'].iloc[-1]:.2f}")
    metric_3.metric("Reward terbaik", f"{history['Reward'].max():.2f}")

    chart_data = history.set_index("Episode")[["Reward", "Rata-rata Reward 40 Episode"]]
    st.line_chart(chart_data)

    with st.expander("Tabel riwayat reward"):
        st.dataframe(history, use_container_width=True, hide_index=True)

    if not losses.empty:
        with st.expander("Grafik loss selama training"):
            st.line_chart(losses.set_index("Step")[["Actor Loss", "Critic Loss"]])
            st.dataframe(losses.tail(20), use_container_width=True, hide_index=True)

    st.subheader("🎮 Uji Agen")
    demo_col_1, demo_col_2 = st.columns([0.35, 0.65])
    with demo_col_1:
        demo_seed = st.number_input("Seed demo", value=123, step=1)
        demo_steps = st.slider("Step demo", 50, 300, 200, 10)
        run_demo = st.button("Jalankan Demo", use_container_width=True)

    if run_demo:
        frames, total_reward, actions = run_agent_demo(agent, seed=int(demo_seed), max_steps=int(demo_steps))
        with demo_col_1:
            st.metric("Total reward demo", f"{total_reward:.2f}")
            if actions:
                st.metric("Rata-rata aksi", f"{np.mean(actions):.3f}")

        with demo_col_2:
            gif_path = make_gif(frames)
            if gif_path:
                st.image(gif_path, caption="Simulasi agen setelah training")
            elif frames:
                st.image(frames[-1], caption="Frame terakhir simulasi")
            else:
                st.warning("Frame simulasi tidak tersedia di perangkat ini.")

        with st.expander("Data aksi pada demo"):
            demo_df = pd.DataFrame({"Step": np.arange(1, len(actions) + 1), "Aksi": actions})
            st.line_chart(demo_df.set_index("Step"))
            st.dataframe(demo_df, use_container_width=True, hide_index=True)

    st.subheader("💾 Simpan Bobot Model")
    weights_zip = make_weights_zip(agent)
    st.download_button(
        label="Download bobot model (.zip)",
        data=weights_zip,
        file_name="ddpg_pendulum_weights.zip",
        mime="application/zip",
        use_container_width=True,
    )
else:
    st.info("Klik tombol **Mulai Training** untuk menampilkan grafik dan hasil evaluasi.")

st.divider()
st.caption("Catatan: hasil training dapat berbeda tergantung seed, jumlah episode, dan kemampuan perangkat yang digunakan.")
