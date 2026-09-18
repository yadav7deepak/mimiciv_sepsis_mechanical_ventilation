"""
D3QN Reinforcement Learning for Mechanical Ventilation (converted from Vent_D3QN.ipynb).
Implements:
1. Dueling Double Deep Q-Network (D3QN) architecture with PyTorch.
2. Experience Replay Buffer and soft target network updates.
3. MDP environment data construction from clustered patient states.
4. Epsilon-greedy training loop with gradient clipping and loss tracking.
5. Greedy policy evaluation and suggested action extraction.
6. Comparison of D3QN agent recommendations vs original physician actions.
7. Saves model weights, comparison table, and publication-grade visualization to ./data/.

Reads preprocessed data from LOCAL_DATA_DIR (./data) with Drive fallback.
Saves outputs locally to LOCAL_DATA_DIR (./data).
"""

import os
import sys
import time
import random
import argparse
from collections import deque
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

# Import configuration and drive utilities
try:
    from .config import (
        LOCAL_DATA_DIR,
        get_output_path,
        resolve_intermediate_path,
    )
    from .drive_utils import mount_drive
except ImportError:
    from config import (
        LOCAL_DATA_DIR,
        get_output_path,
        resolve_intermediate_path,
    )
    from drive_utils import mount_drive


# ==============================================================================
# 1. Reward Function
# ==============================================================================
def get_reward(subject_has_deathtime: int) -> float:
    """Returns -1 for mortality (deathtime present), +1 for survival."""
    return -1.0 if subject_has_deathtime == 1 else 1.0


# ==============================================================================
# 2. Dueling Q-Network (Dueling + Double Q)
# ==============================================================================
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(DuelingQNetwork, self).__init__()

        # Shared feature representations
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # State-value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        V = self.value_stream(features)
        A = self.advantage_stream(features)
        # Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
        Q = V + (A - A.mean(dim=-1, keepdim=True))
        return Q


# ==============================================================================
# 3. Experience Replay Buffer
# ==============================================================================
class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# 4. D3QN Agent
# ==============================================================================
class D3QNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.0005,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: Optional[str] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax().item()

    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train_step(self, batch_size: int = 64) -> Optional[float]:
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        q_values = self.q_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: actions chosen by q_net, evaluated by target_net
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states)
            target_q = rewards + self.gamma * next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1) * (1.0 - dones)

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.soft_update()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()


# ==============================================================================
# 5. Training Loop
# ==============================================================================
# 5. Training Loop
# ==============================================================================
def train_d3qn(
    env_data: List[Dict],
    agent: D3QNAgent,
    epochs: int = 2,
    batch_size: int = 64,
    update_freq: int = 4,
    max_transitions: Optional[int] = None
):
    total_data = len(env_data)
    if max_transitions and max_transitions < total_data:
        total_data = max_transitions

    print(f"\n[vent_d3qn] Starting D3QN training for {epochs} epochs on {total_data:,} transitions (update every {update_freq} transitions)...")
    agent.q_net.train()

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        t_epoch_start = time.time()

        for i in range(total_data - 1):
            state = env_data[i]["state"]
            action = env_data[i]["action_id"]
            reward = env_data[i]["reward"]
            next_state = env_data[i + 1]["state"]
            done = env_data[i]["done"]

            agent.replay_buffer.push(state, action, reward, next_state, done)

            if i % update_freq == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    total_loss += loss
                    count += 1

            if (i + 1) % 200000 == 0 or (i + 1) == total_data - 1:
                curr_avg = total_loss / max(count, 1)
                elapsed = time.time() - t_epoch_start
                print(f"  Epoch {epoch + 1}/{epochs} | Step {i + 1:,}/{total_data:,} | Avg Loss: {curr_avg:.4f} | Epsilon: {agent.epsilon:.3f} | Elapsed: {elapsed:.1f}s")

        avg_loss = total_loss / max(count, 1)
        epoch_time = time.time() - t_epoch_start
        print(f"--> Epoch {epoch + 1:2d}/{epochs} complete in {epoch_time:.1f}s | Avg Loss: {avg_loss:.4f} | Final Epsilon: {agent.epsilon:.3f}")

    print("[vent_d3qn] Training loop finished.")


# ==============================================================================
# 6. Build MDP Environment Dataset
# ==============================================================================
def build_environment_data(
    df: pd.DataFrame,
    df_action_space: pd.DataFrame
) -> Tuple[List[Dict], List[str], D3QNAgent, np.ndarray]:
    """Prepares one-hot encoded state space, maps actions, and constructs transition list."""
    print("\n[vent_d3qn] Constructing MDP environment transition dataset...")
    df_work = df.copy()

    # One-hot encode cluster labels for state space
    if 'cluster_label' in df_work.columns:
        df_work = pd.get_dummies(df_work, columns=['cluster_label'], prefix='cluster_oh')

    state_cols = [c for c in df_work.columns if c.startswith('cluster_oh_')]
    if not state_cols:
        raise ValueError("No 'cluster_oh_' columns found. Please ensure cluster_label exists.")

    merge_cols = [c for c in ['PEEP_binned', 'FiO2_binned', 'tidal_volume_binned', 'RespRate_binned'] if c in df_work.columns and c in df_action_space.columns]
    if 'action_id' not in df_work.columns:
        df_work = pd.merge(df_work, df_action_space, on=merge_cols, how='left')

    df_work['action_id'] = df_work['action_id'].fillna(0).astype(int)

    if 'done_flag' not in df_work.columns:
        df_work['done_flag'] = (df_work.groupby('subject_id').cumcount(ascending=False) == 0).astype(int)

    if 'subject_has_deathtime' not in df_work.columns:
        if 'deathtime' in df_work.columns:
            subjects_with_death = df_work.groupby('subject_id')['deathtime'].apply(lambda x: x.notna().any())
            df_work['subject_has_deathtime'] = df_work['subject_id'].map(subjects_with_death).astype(int)
        elif 'hospmort90day' in df_work.columns:
            df_work['subject_has_deathtime'] = df_work['hospmort90day'].fillna(0).astype(int)
        else:
            df_work['subject_has_deathtime'] = 0

    state_matrix = df_work[state_cols].values.astype(np.float32)
    actions = df_work['action_id'].values
    dones = df_work['done_flag'].values
    deaths = df_work['subject_has_deathtime'].values

    env_data = []
    for i in range(len(df_work)):
        env_data.append({
            "state": state_matrix[i],
            "action_id": int(actions[i]),
            "reward": get_reward(int(deaths[i])),
            "done": int(dones[i])
        })

    state_dim = len(state_cols)
    action_dim = len(df_action_space)
    print(f"MDP constructed: {len(env_data):,} transitions | State Dim: {state_dim} | Action Dim: {action_dim}")

    agent = D3QNAgent(state_dim=state_dim, action_dim=action_dim)
    return env_data, state_cols, agent, state_matrix


# ==============================================================================
# 7. Evaluation & Comparison Visualization
# ==============================================================================
def evaluate_and_compare(
    agent: D3QNAgent,
    env_data: List[Dict],
    df_action_space: pd.DataFrame,
    df_original: pd.DataFrame,
    output_path: str,
    state_matrix: Optional[np.ndarray] = None
):
    """Predicts optimal actions using greedy policy and visualizes vs physician practice."""
    print("\n[vent_d3qn] Evaluating agent performance & episode rewards...")
    agent.q_net.eval()
    orig_eps = agent.epsilon
    agent.epsilon = 0.0

    # 1. Episode reward evaluation (matching notebook Cell 20)
    total_rewards = []
    current_episode_reward = 0.0
    for i in range(len(env_data)):
        reward = env_data[i]["reward"]
        done = env_data[i]["done"]
        current_episode_reward += reward
        if done:
            total_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
    if current_episode_reward != 0.0:
        total_rewards.append(current_episode_reward)

    print(f"  Number of completed episodes: {len(total_rewards):,}")
    if total_rewards:
        print(f"  Average total reward per episode: {np.mean(total_rewards):.2f}")
        print(f"  Min total reward per episode: {np.min(total_rewards):.2f}")
        print(f"  Max total reward per episode: {np.max(total_rewards):.2f}")

    # 2. Vectorized batch policy prediction
    print("\n[vent_d3qn] Predicting optimal actions via greedy policy (batch inference)...")
    predicted_action_ids = []
    with torch.no_grad():
        if state_matrix is not None:
            for start_idx in range(0, len(state_matrix), 8192):
                batch = torch.from_numpy(state_matrix[start_idx:start_idx + 8192]).float().to(agent.device)
                preds = agent.q_net(batch).argmax(dim=1).cpu().numpy()
                predicted_action_ids.extend(preds.tolist())
        else:
            states_arr = np.array([item["state"] for item in env_data], dtype=np.float32)
            for start_idx in range(0, len(states_arr), 8192):
                batch = torch.from_numpy(states_arr[start_idx:start_idx + 8192]).float().to(agent.device)
                preds = agent.q_net(batch).argmax(dim=1).cpu().numpy()
                predicted_action_ids.extend(preds.tolist())

    agent.epsilon = orig_eps
    agent.q_net.train()

    # Map action IDs back to discrete bins
    action_lookup = df_action_space.set_index('action_id')
    predicted_actions_df = action_lookup.loc[predicted_action_ids].reset_index(drop=True)

    binned_vars = [c for c in ['PEEP_binned', 'FiO2_binned', 'tidal_volume_binned', 'RespRate_binned'] if c in df_original.columns and c in predicted_actions_df.columns]

    comparison_records = []
    for var in binned_vars:
        agent_counts = predicted_actions_df[var].value_counts()
        physician_counts = df_original[var].value_counts()
        all_bins = sorted(set(agent_counts.index).union(set(physician_counts.index)))

        for b in all_bins:
            comparison_records.append({
                'Variable': var.replace('_binned', ''),
                'Bin': b,
                'Agent_Count': int(agent_counts.get(b, 0)),
                'Physician_Count': int(physician_counts.get(b, 0))
            })

    comparison_df = pd.DataFrame(comparison_records)
    csv_out = os.path.join(output_path, "action_distribution_comparison.csv")
    comparison_df.to_csv(csv_out, index=False)
    print(f"[vent_d3qn] Comparison table saved to: {csv_out}")

    # Generate research-grade visualization
    try:
        df_melted = comparison_df.melt(
            id_vars=['Variable', 'Bin'],
            value_vars=['Agent_Count', 'Physician_Count'],
            var_name='Source',
            value_name='Count'
        )
        df_melted['Source'] = df_melted['Source'].replace({
            'Agent_Count': 'D3QN Agent',
            'Physician_Count': 'Physician (Actual)'
        })

        variables = df_melted['Variable'].unique()
        fig, axes = plt.subplots(len(variables), 1, figsize=(10, 3.5 * len(variables)), constrained_layout=True)
        if len(variables) == 1:
            axes = [axes]

        palette = {'D3QN Agent': '#1f77b4', 'Physician (Actual)': '#ff7f0e'}

        for i, var in enumerate(variables):
            sub_df = df_melted[df_melted['Variable'] == var]
            sns.barplot(data=sub_df, x='Bin', y='Count', hue='Source', ax=axes[i], palette=palette)
            axes[i].set_title(f"Action Distribution Comparison: {var}", fontsize=14, fontweight='bold')
            axes[i].set_xlabel("Discretized Bin", fontsize=12)
            axes[i].set_ylabel("Observation Count", fontsize=12)
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)

        plot_file = os.path.join(output_path, "action_distribution_comparison.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"[vent_d3qn] Distribution comparison plot saved to: {plot_file}")
    except Exception as e:
        print(f"[vent_d3qn] Warning: Error generating plot: {e}")


def run_all(
    output_path: str = str(LOCAL_DATA_DIR),
    epochs: int = 2,
    batch_size: int = 64,
    update_freq: int = 4,
    max_transitions: Optional[int] = None
):
    """Executes full D3QN RL pipeline."""
    mount_drive()
    os.makedirs(output_path, exist_ok=True)

    cluster_file = resolve_intermediate_path("final_patient_dataset_mdp.csv")
    action_space_file = resolve_intermediate_path("action_space.csv")

    print(f"Loading clustered dataset: {cluster_file}")
    df_cluster = pd.read_csv(cluster_file)
    print(f"Loading action space: {action_space_file}")
    df_action_space = pd.read_csv(action_space_file)

    env_data, state_cols, agent, state_matrix = build_environment_data(df_cluster, df_action_space)
    train_d3qn(
        env_data,
        agent,
        epochs=epochs,
        batch_size=batch_size,
        update_freq=update_freq,
        max_transitions=max_transitions
    )

    # Save model weights
    model_save_path = os.path.join(output_path, "d3qn_model.pth")
    torch.save(agent.q_net.state_dict(), model_save_path)
    print(f"[vent_d3qn] Model weights saved to: {model_save_path}")

    # Evaluate and visualize
    evaluate_and_compare(agent, env_data, df_action_space, df_cluster, output_path, state_matrix=state_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate D3QN Ventilation Agent")
    parser.add_argument("--output-dir", type=str, default=str(LOCAL_DATA_DIR), help="Local output directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for ReplayBuffer")
    parser.add_argument("--update-freq", type=int, default=4, help="Frequency of gradient updates per transition")
    parser.add_argument("--max-transitions", type=int, default=None, help="Optional limit on transitions for fast testing")
    args = parser.parse_args()

    run_all(
        output_path=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        update_freq=args.update_freq,
        max_transitions=args.max_transitions
    )
