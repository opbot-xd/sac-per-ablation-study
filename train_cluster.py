#!/usr/bin/env python
# coding: utf-8

# # SAC vs SACPER — Rigorous Ablation Study on ANM6Easy-v0
# ## Full experimental suite: multi-seed, priority analysis, N-step returns, Q-bias, reward shaping
# 
# **Research questions answered here:**
# 1. Is SAC vs SACPER performance difference due to PER, or due to co-bundled stability fixes?
# 2. How sensitive is PER to the priority exponent α (0.3 vs 0.6)?
# 3. Do N-step returns (n=3,5) improve sample efficiency on top of PER?
# 4. Is the λ-curriculum compatible with PER, or fundamentally incompatible?
# 5. Can running reward normalisation replace the λ-curriculum without PER?
# 6. Do critics overestimate Q-values, and does PER make it worse?
# 7. How does the priority distribution evolve during training?
# 
# **Experiment matrix:**
# 
# | Variant | Replay | α exponent | N-step | λ-curriculum | Reward norm | Grad clip | Warmup |
# |---|---|---|---|---|---|---|---|
# | `SAC_orig` | Uniform 300k | — | 1 | ✓ (10→1000) | ✗ | 5.0 | 10k |
# | `SAC_fixed` | Uniform 500k | — | 1 | ✗ | ✗ | 1.0 | 20k |
# | `SACPER_a03` | PER 500k | **0.3** | 1 | ✗ | ✗ | 1.0 | 20k |
# | `SACPER_a06` | PER 500k | **0.6** | 1 | ✗ | ✗ | 1.0 | 20k |
# | `SACPER_n3` | PER 500k | 0.3 | **3** | ✗ | ✗ | 1.0 | 20k |
# | `SACPER_n5` | PER 500k | 0.3 | **5** | ✗ | ✗ | 1.0 | 20k |
# | `SAC_rewnorm` | Uniform 500k | — | 1 | ✗ | **RunMeanStd** | 1.0 | 20k |
# | `SACPER_lam` | PER 500k | 0.3 | 1 | **✓ (slow ramp)** | ✗ | 1.0 | 20k |
# 
# Each variant runs with **N_SEEDS = 3** seeds (42, 123, 456).  
# All plots show mean ± shaded 95 % CI across seeds.
# 

# In[1]:


# ── Install / upgrade dependencies ──────────────────────────────────────────
# Uncomment if running fresh on Colab
# !pip install gymnasium gym-anm cpprb torch numpy matplotlib scipy --quiet
import sys, subprocess
try:
    import cpprb
    print("cpprb already installed")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "cpprb", "--quiet"])
print("All dependencies ready.")


# In[2]:


import gymnasium as gym
import gym_anm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from dataclasses import dataclass, field
from typing import List
import random, collections, pickle, os, time, math, copy, warnings
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
warnings.filterwarnings("ignore")

try:
    from cpprb import PrioritizedReplayBuffer as CppPER
    HAS_CPPRB = True
    print("cpprb found — C++ PER backend available")
except ImportError:
    HAS_CPPRB = False
    print("cpprb not found — using pure-Python PER (slower but works)")

torch.set_flush_denormal(True)
DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Probe environment dims once
_env = gym.make("ANM6Easy-v0")
_obs, _ = _env.reset(seed=0)
STATE_DIM  = _obs.shape[0]
ACTION_DIM = _env.action_space.shape[0]
ACTION_LOW  = _env.action_space.low.copy()
ACTION_HIGH = _env.action_space.high.copy()
_env.close()
print(f"State={STATE_DIM}, Action={ACTION_DIM}")


# In[3]:


@dataclass
class ExpConfig:
    name:               str
    use_per:            bool  = False
    per_alpha:          float = 0.3
    per_beta_start:     float = 0.4
    per_beta_end:       float = 1.0
    grad_clip:          float = 1.0
    warmup_steps:       int   = 20_000
    use_lambda:         bool  = False
    lambda_start:       float = 10.0
    lambda_end:         float = 1000.0
    lambda_ramp_steps:  int   = 150_000
    use_reward_norm:    bool  = False
    buffer_size:        int   = 500_000
    n_step:             int   = 1
    lr_alpha:           float = 1e-5
    update_every:       int   = 2
    total_steps:        int   = 300_000
    seeds:              List[int] = field(default_factory=lambda: [42, 123, 456])

    def gamma_n(self, gamma: float) -> float:
        return gamma ** self.n_step

EXPERIMENTS: dict[str, ExpConfig] = {
    "SAC_orig": ExpConfig(
        name="SAC_orig",
        use_per=False, grad_clip=5.0, warmup_steps=10_000,
        use_lambda=True, buffer_size=300_000,
        lr_alpha=3e-5, update_every=1
    ),
    "SAC_fixed": ExpConfig(
        name="SAC_fixed",
        use_per=False, grad_clip=1.0, warmup_steps=20_000,
        use_lambda=False, buffer_size=500_000,
        lr_alpha=1e-5, update_every=2
    ),
    "SACPER_a03": ExpConfig(
        name="SACPER_a03",
        use_per=True, per_alpha=0.3, grad_clip=1.0, warmup_steps=20_000,
        use_lambda=False, buffer_size=500_000,
        lr_alpha=1e-5, update_every=2
    ),
    "SACPER_a06": ExpConfig(
        name="SACPER_a06",
        use_per=True, per_alpha=0.6, grad_clip=1.0, warmup_steps=20_000,
        use_lambda=False, buffer_size=500_000,
        lr_alpha=1e-5, update_every=2
    ),
    "SACPER_n3": ExpConfig(
        name="SACPER_n3",
        use_per=True, per_alpha=0.3, n_step=3, grad_clip=1.0, warmup_steps=20_000,
        use_lambda=False, buffer_size=500_000,
        lr_alpha=1e-5, update_every=2
    ),
    "SACPER_n5": ExpConfig(
        name="SACPER_n5",
        use_per=True, per_alpha=0.3, n_step=5, grad_clip=1.0, warmup_steps=20_000,
        use_lambda=False, buffer_size=500_000,
        lr_alpha=1e-5, update_every=2
    ),
    "SAC_rewnorm": ExpConfig(
        name="SAC_rewnorm",
        use_per=False, use_reward_norm=True, grad_clip=1.0, warmup_steps=20_000,
        use_lambda=False, buffer_size=500_000,
        lr_alpha=1e-5, update_every=2
    ),
    "SACPER_lam": ExpConfig(
        name="SACPER_lam",
        use_per=True, per_alpha=0.3, grad_clip=1.0, warmup_steps=20_000,
        use_lambda=True, lambda_ramp_steps=250_000,  # slower ramp for PER compatibility
        buffer_size=500_000, lr_alpha=1e-5, update_every=2
    ),
}

# ── Shared hyperparameters (fixed across all experiments) ────────────────────
GAMMA        = 0.995
TAU          = 1e-2
LR_ACTOR     = 1e-3
LR_CRITIC    = 1e-3
HIDDEN       = 256
N_HIDDEN     = 2
BATCH_SIZE   = 256
INIT_ALPHA   = 0.2
ALPHA_FLOOR  = 0.05
R_CLIP       = 100.0
LOG_STD_MIN  = -20
LOG_STD_MAX  = 2
ZETA         = 1e-6            # PER priority floor
EVAL_EVERY   = 20_000
N_EVAL       = 3               # rollouts per evaluation
EVAL_HORIZON = 1_000
EPISODE_LEN  = 5_000
MPC_BASELINE = -129.1          # published MPC score to beat

print("Config dataclass and experiment matrix defined.")
for k, v in EXPERIMENTS.items():
    print(f"  {k:15s}  per={v.use_per}  α={v.per_alpha if v.use_per else '—':>4}  n={v.n_step}  λ={v.use_lambda}  rnorm={v.use_reward_norm}")


# In[4]:


# ── Network architectures (shared across all variants) ──────────────────────

def mlp(in_dim, out_dim, hidden=HIDDEN, n_layers=N_HIDDEN, act=nn.ReLU):
    layers, dim = [], in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(dim, hidden), act()]
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, sd, ad, scale, bias):
        super().__init__()
        self.net = mlp(sd, hidden=HIDDEN, out_dim=HIDDEN)
        self.mu  = nn.Linear(HIDDEN, ad)
        self.ls  = nn.Linear(HIDDEN, ad)
        self.sc  = torch.FloatTensor(scale).to(DEVICE)
        self.bi  = torch.FloatTensor(bias).to(DEVICE)

    def forward(self, s):
        x = self.net(s)
        return self.mu(x), self.ls(x).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, s):
        mu, ls = self.forward(s)
        std = ls.exp()
        d   = Normal(mu, std)
        u   = d.rsample()
        a   = torch.tanh(u)
        lp  = (d.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a * self.sc + self.bi, lp, torch.tanh(mu) * self.sc + self.bi


class Critic(nn.Module):
    def __init__(self, sd, ad):
        super().__init__()
        self.q1 = mlp(sd + ad, out_dim=1)
        self.q2 = mlp(sd + ad, out_dim=1)

    def forward(self, s, a):
        sa = torch.cat([s, a], -1)
        return self.q1(sa), self.q2(sa)


print("Actor and Critic defined.")


# In[5]:


# ── Replay Buffers: Uniform, PER (Python), N-step wrapper ───────────────────

class UniformBuffer:
    def __init__(self, cap):
        self.buf = collections.deque(maxlen=cap)

    def push(self, s, a, r, s_, d):
        self.buf.append((s, a, r, s_, d))

    def sample(self, n):
        s, a, r, s_, d = zip(*random.sample(self.buf, n))
        t = lambda x: torch.FloatTensor(np.array(x)).to(DEVICE)
        return (t(s), t(a),
                t(r).unsqueeze(1),
                t(s_),
                t(d).unsqueeze(1),
                torch.ones(n, 1, device=DEVICE),   # IS weights = 1 for uniform
                None)                               # no indexes to update

    def __len__(self): return len(self.buf)
    def get_stored_size(self): return len(self.buf)

    # Priority analysis: not applicable
    def get_priority_stats(self):
        return None


class PERBuffer:
    """Pure-Python PER — exposes priorities for analysis."""
    def __init__(self, cap, alpha):
        self.cap   = cap
        self.alpha = alpha
        self.buf   = []
        self.prios = np.zeros(cap, dtype=np.float64)
        self.pos   = 0
        self.size  = 0

    def push(self, s, a, r, s_, d):
        max_p = self.prios[:self.size].max() if self.size > 0 else 1.0
        if self.size < self.cap:
            self.buf.append((s, a, r, s_, d))
            self.size += 1
        else:
            self.buf[self.pos] = (s, a, r, s_, d)
        self.prios[self.pos] = max_p
        self.pos = (self.pos + 1) % self.cap

    def sample(self, n, beta):
        p  = self.prios[:self.size] ** self.alpha
        p /= p.sum()
        idxs = np.random.choice(self.size, n, replace=False, p=p)
        w = (self.size * p[idxs]) ** (-beta)
        w = (w / w.max()).astype(np.float32)
        batch = [self.buf[i] for i in idxs]
        s, a, r, s_, d = zip(*batch)
        t = lambda x, dt=np.float32: torch.FloatTensor(np.array(x, dtype=dt)).to(DEVICE)
        ws = torch.FloatTensor(w).unsqueeze(1).to(DEVICE)
        return (t(s), t(a),
                t(r).unsqueeze(1),
                t(s_),
                t(d).unsqueeze(1),
                ws, idxs)

    def update_priorities(self, idxs, td_errors):
        for i, e in zip(idxs, td_errors):
            self.prios[i] = float(abs(e)) + ZETA

    def get_stored_size(self): return self.size

    def get_priority_stats(self):
        """Return Gini coefficient, normalised ESS, normalised entropy."""
        if self.size < 2:
            return dict(gini=0.0, ess_norm=1.0, entropy_norm=1.0)
        p_raw  = self.prios[:self.size] ** self.alpha
        p      = p_raw / p_raw.sum()
        sorted_p = np.sort(p)
        n      = len(sorted_p)
        gini   = (2 * np.dot(np.arange(1, n+1), sorted_p) - (n+1)) / n
        ess    = 1.0 / np.dot(p, p)
        ess_n  = ess / n
        ent    = -np.dot(p, np.log(p + 1e-14))
        ent_n  = ent / np.log(n)
        return dict(gini=float(gini), ess_norm=float(ess_n), entropy_norm=float(ent_n))


class NStepBuffer:
    """Wraps any replay buffer to provide N-step bootstrapped returns."""
    def __init__(self, n: int, gamma: float, replay_buffer):
        self.n     = n
        self.gamma = gamma
        self.replay = replay_buffer
        self.queue: collections.deque = collections.deque()

    # ── Public interface (same as UniformBuffer / PERBuffer) ──────────────
    def push(self, s, a, r, s_, done):
        self.queue.append((s, a, r, s_, done))
        # Commit oldest transition with n-step return whenever queue is full
        while len(self.queue) >= self.n:
            self._commit_front(self.n)
            self.queue.popleft()
        # At episode end, flush shorter returns for remaining transitions
        if done:
            while self.queue:
                self._commit_front(len(self.queue))
                self.queue.popleft()

    def sample(self, *args, **kwargs):
        return self.replay.sample(*args, **kwargs)

    def update_priorities(self, *args, **kwargs):
        return self.replay.update_priorities(*args, **kwargs)

    def get_stored_size(self):
        return self.replay.get_stored_size()

    def get_priority_stats(self):
        return self.replay.get_priority_stats()

    def __len__(self): return len(self.replay) if hasattr(self.replay, '__len__') else self.replay.get_stored_size()

    # ── Internal ───────────────────────────────────────────────────────────
    def _commit_front(self, length):
        q   = list(self.queue)[:length]
        s0, a0 = q[0][0], q[0][1]
        G, last_s_, last_done = 0.0, q[-1][3], q[-1][4]
        for i, (_, _, ri, si_, di) in enumerate(q):
            G += (self.gamma ** i) * ri
            if di:
                last_s_, last_done = si_, True
                break
        self.replay.push(s0, a0, G, last_s_, last_done)


print("UniformBuffer, PERBuffer and NStepBuffer defined.")


# In[6]:


# ── Unified SAC / SACPER Agent ──────────────────────────────────────────────

class SACAgent:
    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        sc = (ACTION_HIGH - ACTION_LOW) / 2.
        bi = (ACTION_HIGH + ACTION_LOW) / 2.

        self.actor    = Actor(STATE_DIM, ACTION_DIM, sc, bi).to(DEVICE)
        self.critic   = Critic(STATE_DIM, ACTION_DIM).to(DEVICE)
        self.critic_t = deepcopy(self.critic)
        for p in self.critic_t.parameters():
            p.requires_grad = False

        self.aopt = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.copt = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.target_ent = -float(ACTION_DIM)
        self.log_alpha  = torch.tensor(math.log(INIT_ALPHA), dtype=torch.float32,
                                       requires_grad=True, device=DEVICE)
        self.lopt = optim.Adam([self.log_alpha], lr=cfg.lr_alpha)

        # Build the correct replay / n-step wrapper
        raw_buf = (PERBuffer(cfg.buffer_size, cfg.per_alpha)
                   if cfg.use_per else UniformBuffer(cfg.buffer_size))
        self.buffer = (NStepBuffer(cfg.n_step, GAMMA, raw_buf)
                       if cfg.n_step > 1 else raw_buf)
        self._raw_buf = raw_buf   # direct access for priority stats

        # γ^n — used in Bellman backup for n-step
        self._gamma_n = cfg.gamma_n(GAMMA)

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def alpha(self):
        return torch.clamp(self.log_alpha.exp(), min=ALPHA_FLOOR).detach()

    # ── Interaction ───────────────────────────────────────────────────────
    def push(self, s, a, r, s_, d):
        self.buffer.push(s, a, r, s_, float(d))

    def buf_len(self):
        return self.buffer.get_stored_size() if hasattr(self.buffer, 'get_stored_size') else len(self.buffer)

    def act(self, s, det=False):
        st = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, _, da = self.actor.sample(st)
            if det:
                return da.cpu().numpy()[0]
            sa, _, _ = self.actor.sample(st)
            return sa.cpu().numpy()[0]

    # ── Update ────────────────────────────────────────────────────────────
    def update(self, step: int):
        if self.buf_len() < BATCH_SIZE:
            return {}

        # Sample
        if self.cfg.use_per:
            beta = (self.cfg.per_beta_start
                    + min(step / self.cfg.total_steps, 1.0)
                    * (self.cfg.per_beta_end - self.cfg.per_beta_start))
            s, a, r, s_, d, w, idxs = self.buffer.sample(BATCH_SIZE, beta=beta)
        else:
            s, a, r, s_, d, w, idxs = self.buffer.sample(BATCH_SIZE)

        al = self.alpha

        # ── Critic update (n-step Bellman) ─────────────────────────────────
        with torch.no_grad():
            an, ln, _    = self.actor.sample(s_)
            q1n, q2n     = self.critic_t(s_, an)
            y = r + self._gamma_n * (1 - d) * (torch.min(q1n, q2n) - al * ln)

        q1, q2  = self.critic(s, a)
        t1, t2  = q1 - y, q2 - y
        c_loss  = (w * t1.pow(2)).mean() + (w * t2.pow(2)).mean()
        self.copt.zero_grad(); c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.copt.step()

        # ── Actor update ───────────────────────────────────────────────────
        an2, ln2, _ = self.actor.sample(s)
        q1b, q2b    = self.critic(s, an2)
        a_loss      = (al * ln2 - torch.min(q1b, q2b)).mean()
        self.aopt.zero_grad(); a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.aopt.step()

        # ── Alpha (entropy temperature) update ────────────────────────────
        alpha_loss = -(self.log_alpha * (ln2.detach() + self.target_ent)).mean()
        self.lopt.zero_grad(); alpha_loss.backward(); self.lopt.step()

        # ── Soft target update ────────────────────────────────────────────
        for tp, p in zip(self.critic_t.parameters(), self.critic.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        # ── Update PER priorities ─────────────────────────────────────────
        if self.cfg.use_per and idxs is not None:
            td = np.clip((t1.abs() + t2.abs()).detach().cpu().numpy().flatten(), 0, 100.0)
            self.buffer.update_priorities(idxs, td)

        return {
            "c_loss":     c_loss.item(),
            "a_loss":     a_loss.item(),
            "alpha":      al.item(),
            "alpha_loss": alpha_loss.item(),
            "td_mean":    ((t1.abs() + t2.abs()) / 2).mean().item(),
        }

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, path, step):
        torch.save({"step": step,
                    "actor":    self.actor.state_dict(),
                    "critic":   self.critic.state_dict(),
                    "critic_t": self.critic_t.state_dict(),
                    "la":       self.log_alpha.data,
                    "aopt":     self.aopt.state_dict(),
                    "copt":     self.copt.state_dict(),
                    "lopt":     self.lopt.state_dict()}, path)

    def load(self, path):
        c = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(c["actor"])
        self.critic.load_state_dict(c["critic"])
        self.critic_t.load_state_dict(c.get("critic_t", c["critic"]))
        self.log_alpha.data.copy_(c["la"])
        for opt, key in [(self.aopt,"aopt"),(self.copt,"copt"),(self.lopt,"lopt")]:
            if key in c: opt.load_state_dict(c[key])
        return c["step"]


print("SACAgent defined.")


# In[7]:


# ── Analysis utilities ──────────────────────────────────────────────────────

class RunningMeanStd:
    """Online normalisation of reward stream (Welford algorithm)."""
    def __init__(self, eps=1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = eps

    def update(self, x):
        b_mean, b_var, b_n = float(x), 0.0, 1
        delta    = b_mean - self.mean
        new_n    = self.count + b_n
        self.mean = self.mean + delta * b_n / new_n
        m_a      = self.var   * self.count
        m_b      = b_var      * b_n
        M2       = m_a + m_b + delta**2 * self.count * b_n / new_n
        self.var = M2 / new_n
        self.count = new_n

    def normalise(self, x):
        return float(x - self.mean) / (math.sqrt(self.var) + 1e-8)


def norm_obs(obs, lo, hi):
    return 2.0 * (obs - lo) / (hi - lo + 1e-8) - 1.0


def get_lambda(step, cfg: ExpConfig):
    if not cfg.use_lambda:
        return 1000.0
    return cfg.lambda_start + min(step / cfg.lambda_ramp_steps, 1.0) * (cfg.lambda_end - cfg.lambda_start)


def shape_reward(rew, step, cfg: ExpConfig, rms: RunningMeanStd = None):
    if cfg.use_reward_norm and rms is not None:
        rms.update(rew)
        rew = rms.normalise(rew) * 10.0   # rescale to ~[-10, 10]
        return np.clip(rew, -R_CLIP, R_CLIP)
    if cfg.use_lambda:
        sc = get_lambda(step, cfg) / 1000.0
        if rew < -10:
            rew = max(rew * sc, rew / 10)
    return float(np.clip(rew, -R_CLIP, R_CLIP))


def iqm(arr):
    """Interquartile mean — robust to extreme outliers (Agarwal et al. 2021)."""
    arr = np.asarray(arr, dtype=float)
    q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
    mid = arr[(arr >= q25) & (arr <= q75)]
    return float(np.mean(mid)) if len(mid) > 0 else float(np.mean(arr))


def evaluate_with_metrics(agent, env, n_rollouts=N_EVAL, horizon=EVAL_HORIZON):
    """
    Returns:
        mean_return, std_return,
        mean_q_bias (predicted Q(s0,a0) − actual discounted return),
        list of per-ep returns
    """
    lo, hi = env.observation_space.low, env.observation_space.high
    returns, q_biases = [], []

    for ep in range(n_rollouts):
        obs, _ = env.reset(seed=ep * 17)
        s0_norm = norm_obs(obs, lo, hi)
        a0      = agent.act(s0_norm, det=True)

        # Predicted Q at (s0, a0)
        with torch.no_grad():
            s_t = torch.FloatTensor(s0_norm).unsqueeze(0).to(DEVICE)
            a_t = torch.FloatTensor(a0).unsqueeze(0).to(DEVICE)
            q1p, q2p = agent.critic(s_t, a_t)
            q_pred = torch.min(q1p, q2p).item()

        G, g = 0.0, 1.0
        done = False
        for _ in range(horizon):
            try:
                obs, r, term, trunc, _ = env.step(agent.act(norm_obs(obs, lo, hi), det=True))
            except RuntimeError:
                r, term = -R_CLIP, True
            G += g * r; g *= GAMMA
            if term or trunc: break

        returns.append(G)
        q_biases.append(q_pred - G)

    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(q_biases)), returns


print("Utilities (RunningMeanStd, IQM, evaluate_with_metrics) defined.")


# In[8]:


# ── Training loop ────────────────────────────────────────────────────────────

def train_one_seed(cfg: ExpConfig, seed: int, results_dir: str = "results") -> dict:
    """
    Train one (config, seed) pair. Returns a log dict and saves it to disk.
    Skips training if a saved log already exists.
    """
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"{cfg.name}_seed{seed}.pkl")

    if os.path.exists(log_path):
        with open(log_path, "rb") as f:
            print(f"  [cache] loaded {log_path}")
            return pickle.load(f)

    print(f"  [train] {cfg.name}  seed={seed}")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    agent = SACAgent(cfg)
    rms   = RunningMeanStd() if cfg.use_reward_norm else None

    log = {
        "steps": [], "returns": [], "returns_std": [], "alpha": [],
        "lambda_val": [], "q_bias": [], "td_mean": [],
        "priority_gini": [], "priority_ess": [], "priority_entropy": [],
        "all_returns": [],   # per-seed eval returns for IQM
    }

    tenv = gym.make("ANM6Easy-v0")
    eenv = gym.make("ANM6Easy-v0")
    lo, hi = tenv.observation_space.low, tenv.observation_space.high
    obs, _ = tenv.reset(seed=seed)

    step = 0; ep_step = 0; crashes = 0
    next_eval = EVAL_EVERY
    t0 = time.time()

    while step < cfg.total_steps:
        s_norm = norm_obs(obs, lo, hi)
        action = (tenv.action_space.sample() if step < cfg.warmup_steps
                  else agent.act(s_norm))

        try:
            nobs, rew, term, trunc, _ = tenv.step(action)
        except RuntimeError:
            crashes += 1
            obs, _ = tenv.reset(seed=seed + step)
            ep_step = 0; step += 1; continue

        ep_step += 1
        done = term or trunc or (ep_step >= EPISODE_LEN)
        rew  = shape_reward(rew, step, cfg, rms)

        agent.push(s_norm, action, rew, norm_obs(nobs, lo, hi), float(term))
        obs = nobs; step += 1

        if done:
            obs, _ = tenv.reset(seed=seed + step)
            ep_step = 0

        # Gradient update
        update_info = {}
        if step >= cfg.warmup_steps and step % cfg.update_every == 0:
            update_info = agent.update(step)

        # ── Evaluation ─────────────────────────────────────────────────────
        if step >= next_eval:
            mr, sr, qb, per_ep_rets = evaluate_with_metrics(agent, eenv)
            lam = get_lambda(step, cfg)

            # Priority distribution
            pstats = agent._raw_buf.get_priority_stats()
            p_gini = pstats["gini"]         if pstats else 0.0
            p_ess  = pstats["ess_norm"]     if pstats else 1.0
            p_ent  = pstats["entropy_norm"] if pstats else 1.0

            log["steps"].append(step)
            log["returns"].append(mr)
            log["returns_std"].append(sr)
            log["alpha"].append(agent.alpha.item())
            log["lambda_val"].append(lam)
            log["q_bias"].append(qb)
            log["td_mean"].append(update_info.get("td_mean", 0.0))
            log["priority_gini"].append(p_gini)
            log["priority_ess"].append(p_ess)
            log["priority_entropy"].append(p_ent)
            log["all_returns"].append(per_ep_rets)

            elapsed = time.time() - t0
            eta = elapsed / step * (cfg.total_steps - step)
            mpc = " ✓ beat MPC!" if mr > MPC_BASELINE else ""
            print(f"    step {step:>7,} | J={mr:>8.1f}±{sr:.1f} "
                  f"| α={agent.alpha.item():.4f} | Qbias={qb:>6.1f} "
                  f"| Gini={p_gini:.3f} | ETA {int(eta)//60}m{mpc}")

            with open(log_path, "wb") as f:
                pickle.dump(log, f)
            next_eval = step + EVAL_EVERY

    tenv.close(); eenv.close()
    print(f"  Finished {cfg.name} seed={seed} in {(time.time()-t0)/60:.1f} min | crashes={crashes}")
    return log


print("train_one_seed() defined.")


# In[ ]:


# ── Multi-seed runner ────────────────────────────────────────────────────────
import multiprocessing
import concurrent.futures

def _train_worker(args):
    name, seed, results_dir = args
    cfg = EXPERIMENTS[name]
    return name, train_one_seed(cfg, seed, results_dir)

def run_all_experiments(exp_names=None, results_dir="results", max_workers=4):
    """
    Run every (experiment, seed) pair in PARALLEL.
    Skips pairs that already have a saved log.
    Returns all_logs: dict[exp_name → list[log_dict per seed]]
    """
    if exp_names is None:
        exp_names = list(EXPERIMENTS.keys())

    tasks = []
    for name in exp_names:
        cfg = EXPERIMENTS[name]
        for seed in cfg.seeds:
            tasks.append((name, seed, results_dir))

    all_logs = {n: [] for n in exp_names}
    total = len(tasks)
    print(f"Running {len(exp_names)} experiment(s) × seeds = {total} total runs.")
    print(f"Parallel mode enabled: running on {max_workers} processes.\n")

    # Use 'spawn' context to avoid CUDA initialization errors or PyTorch memory issues
    ctx = multiprocessing.get_context("spawn")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_train_worker, t) for t in tasks]
        done = 0
        for future in concurrent.futures.as_completed(futures):
            done += 1
            try:
                name, log = future.result()
                all_logs[name].append(log)
                print(f"[{done}/{total}] Finished task and collected results.")
            except Exception as e:
                print(f"Task generated an exception: {e}")

    return all_logs


def load_all_results(exp_names=None, results_dir="results"):
    """Load cached logs without re-training."""
    if exp_names is None:
        exp_names = list(EXPERIMENTS.keys())
    all_logs = {}
    for name in exp_names:
        cfg  = EXPERIMENTS[name]
        logs = []
        for seed in cfg.seeds:
            p = os.path.join(results_dir, f"{name}_seed{seed}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    logs.append(pickle.load(f))
        if logs:
            all_logs[name] = logs
    return all_logs


print("run_all_experiments() and load_all_results() defined.")

# In[ ]:


# ── ▶ RUN EXPERIMENTS ──────────────────────────────────────────────────────
# This cell trains everything.  Re-run freely — cached runs are auto-skipped.
# Now runs in parallel over 4 cores.

import multiprocessing

pass
pass


# In[11]:


# ── Aggregate results across seeds ─────────────────────────────────────────

def aggregate(logs_list):
    """
    Given a list of log dicts (one per seed), return arrays:
      steps, mean_ret, ci_lo, ci_hi, mean_alpha, mean_qbias, mean_gini, mean_ess
    All return-level statistics use SEM-based 95% CI.
    """
    if not logs_list:
        return None
    steps = np.array(logs_list[0]["steps"])
    n_seeds = len(logs_list)

    def ci(key):
        mat = np.array([l[key] for l in logs_list])   # (n_seeds, n_evals)
        m   = mat.mean(0)
        if n_seeds == 1:
            return m, m - np.abs(np.array(logs_list[0]["returns_std"])), m + np.abs(np.array(logs_list[0]["returns_std"]))
        se  = mat.std(0, ddof=1) / math.sqrt(n_seeds)
        t   = stats.t.ppf(0.975, df=n_seeds - 1)
        return m, m - t * se, m + t * se

    mean_r, lo_r, hi_r = ci("returns")
    return dict(
        steps       = steps,
        mean_r      = mean_r,
        lo_r        = lo_r,
        hi_r        = hi_r,
        mean_alpha  = np.array([l["alpha"]         for l in logs_list]).mean(0),
        mean_qbias  = np.array([l["q_bias"]        for l in logs_list]).mean(0),
        mean_gini   = np.array([l["priority_gini"] for l in logs_list]).mean(0),
        mean_ess    = np.array([l["priority_ess"]  for l in logs_list]).mean(0),
        mean_ent    = np.array([l["priority_entropy"] for l in logs_list]).mean(0),
        n_seeds     = n_seeds,
    )


if __name__ == '__main__':
    import multiprocessing
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    multiprocessing.set_start_method('spawn', force=True)
    # Using 20 workers to leave some overhead for the OS and context switching
    # on an i9 with 24 threads and 100GB RAM, this is still extremely fast!
    all_logs = run_all_experiments(max_workers=20)

