#!/usr/bin/env python
# Distributional SAC + PER (DSAC-PER) Ablation Study on ANM6Easy-v0
# 6 variants x 3 seeds = 18 runs
#
# Variants:
#   SAC_base      — scalar critics, uniform replay (baseline)
#   SACPER_base   — scalar critics, PER (reproduces old null result)
#   DSAC_uniform  — quantile critics, uniform replay
#   DSAC_PER      — quantile critics, PER with distributional TD priorities
#   SAC_LAP       — scalar critics, Loss-Adjusted Prioritization
#   DSAC_LAP      — quantile critics, LAP with distributional TD priorities

import gymnasium as gym
import gym_anm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from dataclasses import dataclass, field
from typing import List
import random, collections, pickle, os, time, math
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

torch.set_flush_denormal(True)
DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

_env = gym.make("ANM6Easy-v0")
_obs, _ = _env.reset(seed=0)
STATE_DIM = _obs.shape[0]
ACTION_DIM = _env.action_space.shape[0]
ACTION_LOW = _env.action_space.low.copy()
ACTION_HIGH = _env.action_space.high.copy()
_env.close()
print(f"State={STATE_DIM}, Action={ACTION_DIM}")

# ── Shared hyperparameters ──
GAMMA = 0.995
TAU = 1e-2
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
HIDDEN = 256
N_HIDDEN = 2
BATCH_SIZE = 256
INIT_ALPHA = 0.2
ALPHA_FLOOR = 0.05
R_CLIP = 100.0
LOG_STD_MIN, LOG_STD_MAX = -20, 2
ZETA = 1e-6
EVAL_EVERY = 20_000
N_EVAL = 3
EVAL_HORIZON = 1_000
EPISODE_LEN = 5_000
MPC_BASELINE = -129.1
N_QUANTILES = 25
HUBER_KAPPA = 1.0
LAP_ETA = 0.6
BUS_LOG_EVERY = 500        # snapshot bus data from training env every N steps


@dataclass
class ExpConfig:
    name: str
    use_distributional: bool = False
    use_per: bool = False
    use_lap: bool = False
    per_alpha: float = 0.3
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    grad_clip: float = 1.0
    warmup_steps: int = 20_000
    buffer_size: int = 500_000
    lr_alpha: float = 1e-5
    update_every: int = 2
    total_steps: int = 300000
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


EXPERIMENTS = {
    "SAC_base": ExpConfig(
        name="SAC_base", use_distributional=False, use_per=False
    ),
    "SACPER_base": ExpConfig(
        name="SACPER_base", use_distributional=False, use_per=True
    ),
    "DSAC_uniform": ExpConfig(
        name="DSAC_uniform", use_distributional=True, use_per=False
    ),
    "DSAC_PER": ExpConfig(
        name="DSAC_PER", use_distributional=True, use_per=True
    ),
    "SAC_LAP": ExpConfig(
        name="SAC_LAP", use_distributional=False, use_per=True, use_lap=True
    ),
    "DSAC_LAP": ExpConfig(
        name="DSAC_LAP", use_distributional=True, use_per=True, use_lap=True
    ),
}

print("Experiment matrix:")
for k, v in EXPERIMENTS.items():
    print(f"  {k:15s}  dist={v.use_distributional}  per={v.use_per}  lap={v.use_lap}")


# ── Networks ──

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
        self.mu = nn.Linear(HIDDEN, ad)
        self.ls = nn.Linear(HIDDEN, ad)
        self.sc = torch.FloatTensor(scale).to(DEVICE)
        self.bi = torch.FloatTensor(bias).to(DEVICE)

    def forward(self, s):
        x = self.net(s)
        return self.mu(x), self.ls(x).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, s):
        mu, ls = self.forward(s)
        std = ls.exp()
        d = Normal(mu, std)
        u = d.rsample()
        a = torch.tanh(u)
        lp = (d.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a * self.sc + self.bi, lp, torch.tanh(mu) * self.sc + self.bi


class ScalarCritic(nn.Module):
    def __init__(self, sd, ad):
        super().__init__()
        self.q1 = mlp(sd + ad, out_dim=1)
        self.q2 = mlp(sd + ad, out_dim=1)

    def forward(self, s, a):
        sa = torch.cat([s, a], -1)
        return self.q1(sa), self.q2(sa)


class QuantileCritic(nn.Module):
    """Outputs N_QUANTILES quantile values for each of two Q-networks."""
    def __init__(self, sd, ad, n_quantiles=N_QUANTILES):
        super().__init__()
        self.n_q = n_quantiles
        self.q1 = mlp(sd + ad, out_dim=n_quantiles)
        self.q2 = mlp(sd + ad, out_dim=n_quantiles)
        # Fixed quantile midpoints: tau_i = (2i+1)/(2N)
        taus = (2 * torch.arange(n_quantiles).float() + 1) / (2 * n_quantiles)
        self.register_buffer("taus", taus)

    def forward(self, s, a):
        sa = torch.cat([s, a], -1)
        return self.q1(sa), self.q2(sa)  # each (B, N_QUANTILES)

    def q_mean(self, s, a):
        """Mean Q-value (average over quantiles)."""
        z1, z2 = self.forward(s, a)
        return z1.mean(-1, keepdim=True), z2.mean(-1, keepdim=True)


# ── Replay Buffers ──

class UniformBuffer:
    def __init__(self, cap):
        self.buf = collections.deque(maxlen=cap)

    def push(self, s, a, r, s_, d):
        self.buf.append((s, a, r, s_, d))

    def sample(self, n):
        s, a, r, s_, d = zip(*random.sample(self.buf, n))
        t = lambda x: torch.FloatTensor(np.array(x)).to(DEVICE)
        return (t(s), t(a), t(r).unsqueeze(1), t(s_), t(d).unsqueeze(1),
                torch.ones(n, 1, device=DEVICE), None)

    def __len__(self):
        return len(self.buf)

    def get_stored_size(self):
        return len(self.buf)

    def get_priority_stats(self):
        return None


class PERBuffer:
    def __init__(self, cap, alpha, use_lap=False, lap_eta=LAP_ETA):
        self.cap = cap
        self.alpha = alpha
        self.use_lap = use_lap
        self.lap_eta = lap_eta
        self.buf = []
        self.prios = np.zeros(cap, dtype=np.float64)
        self.pos = 0
        self.size = 0

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
        p = self.prios[:self.size] ** self.alpha
        p /= p.sum()
        idxs = np.random.choice(self.size, n, replace=False, p=p)
        w = (self.size * p[idxs]) ** (-beta)
        w = (w / w.max()).astype(np.float32)
        batch = [self.buf[i] for i in idxs]
        s, a, r, s_, d = zip(*batch)
        t = lambda x: torch.FloatTensor(np.array(x, dtype=np.float32)).to(DEVICE)
        ws = torch.FloatTensor(w).unsqueeze(1).to(DEVICE)
        return (t(s), t(a), t(r).unsqueeze(1), t(s_), t(d).unsqueeze(1), ws, idxs)

    def update_priorities(self, idxs, td_errors):
        max_p = self.prios[:self.size].max() if self.size > 0 else 1.0
        for i, e in zip(idxs, td_errors):
            raw = float(abs(e)) + ZETA
            if self.use_lap:
                self.prios[i] = self.lap_eta * max_p + (1 - self.lap_eta) * raw
            else:
                self.prios[i] = raw

    def get_stored_size(self):
        return self.size

    def get_priority_stats(self):
        if self.size < 2:
            return dict(gini=0.0, ess_norm=1.0, entropy_norm=1.0)
        p_raw = self.prios[:self.size] ** self.alpha
        p = p_raw / p_raw.sum()
        sorted_p = np.sort(p)
        n = len(sorted_p)
        gini = (2 * np.dot(np.arange(1, n + 1), sorted_p) - (n + 1)) / n
        ess = 1.0 / np.dot(p, p)
        ess_n = ess / n
        ent = -np.dot(p, np.log(p + 1e-14))
        ent_n = ent / np.log(n)
        return dict(gini=float(gini), ess_norm=float(ess_n), entropy_norm=float(ent_n))


# ── Quantile Huber Loss ──

def quantile_huber_loss(quantiles_pred, target, taus, kappa=HUBER_KAPPA):
    """
    quantiles_pred: (B, N_Q) predicted quantile values
    target: (B, 1) scalar target (will be broadcast)
    taus: (N_Q,) quantile midpoints
    Returns: loss (scalar), per-sample mean |TD| (B,)
    """
    # (B, N_Q) element-wise TD error
    td = target - quantiles_pred
    abs_td = td.abs()
    huber = torch.where(abs_td <= kappa,
                        0.5 * td.pow(2),
                        kappa * (abs_td - 0.5 * kappa))
    # Asymmetric weighting by quantile level
    weight = (taus.unsqueeze(0) - (td < 0).float()).abs()
    loss = (weight * huber).mean()
    return loss, abs_td.mean(dim=1)  # loss, per-sample priority signal


# ── Unified Agent ──

class DSACAgent:
    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        sc = (ACTION_HIGH - ACTION_LOW) / 2.0
        bi = (ACTION_HIGH + ACTION_LOW) / 2.0

        self.actor = Actor(STATE_DIM, ACTION_DIM, sc, bi).to(DEVICE)

        if cfg.use_distributional:
            self.critic = QuantileCritic(STATE_DIM, ACTION_DIM).to(DEVICE)
        else:
            self.critic = ScalarCritic(STATE_DIM, ACTION_DIM).to(DEVICE)

        self.critic_t = deepcopy(self.critic)
        for p in self.critic_t.parameters():
            p.requires_grad = False

        self.aopt = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.copt = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.target_ent = -float(ACTION_DIM)
        self.log_alpha = torch.tensor(
            math.log(INIT_ALPHA), dtype=torch.float32,
            requires_grad=True, device=DEVICE
        )
        self.lopt = optim.Adam([self.log_alpha], lr=cfg.lr_alpha)

        if cfg.use_per:
            raw_buf = PERBuffer(cfg.buffer_size, cfg.per_alpha,
                                use_lap=cfg.use_lap)
        else:
            raw_buf = UniformBuffer(cfg.buffer_size)
        self.buffer = raw_buf
        self._raw_buf = raw_buf

    @property
    def alpha(self):
        return torch.clamp(self.log_alpha.exp(), min=ALPHA_FLOOR).detach()

    def push(self, s, a, r, s_, d):
        self.buffer.push(s, a, r, s_, float(d))

    def buf_len(self):
        return self.buffer.get_stored_size()

    def act(self, s, det=False):
        st = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, _, da = self.actor.sample(st)
            if det:
                return da.cpu().numpy()[0]
            sa, _, _ = self.actor.sample(st)
            return sa.cpu().numpy()[0]

    def update(self, step):
        if self.buf_len() < BATCH_SIZE:
            return {}

        if self.cfg.use_per:
            beta = (self.cfg.per_beta_start
                    + min(step / self.cfg.total_steps, 1.0)
                    * (self.cfg.per_beta_end - self.cfg.per_beta_start))
            s, a, r, s_, d, w, idxs = self.buffer.sample(BATCH_SIZE, beta=beta)
        else:
            s, a, r, s_, d, w, idxs = self.buffer.sample(BATCH_SIZE)

        al = self.alpha

        # ── Critic update ──
        with torch.no_grad():
            an, ln, _ = self.actor.sample(s_)
            if self.cfg.use_distributional:
                q1n, q2n = self.critic_t(s_, an)  # (B, N_Q)
                q1m, q2m = q1n.mean(-1, keepdim=True), q2n.mean(-1, keepdim=True)
                # Use quantiles from the network with lower mean
                mask = (q1m <= q2m).float()
                qn = mask * q1n + (1 - mask) * q2n  # (B, N_Q)
                y = r + GAMMA * (1 - d) * (qn - al * ln)  # (B, N_Q)
            else:
                q1n, q2n = self.critic_t(s_, an)
                y = r + GAMMA * (1 - d) * (torch.min(q1n, q2n) - al * ln)

        if self.cfg.use_distributional:
            q1, q2 = self.critic(s, a)  # (B, N_Q)
            # Target for each quantile: y is (B, N_Q)
            # Quantile Huber loss for each critic
            taus = self.critic.taus
            td1 = y.detach() - q1  # (B, N_Q)
            td2 = y.detach() - q2
            abs_td1, abs_td2 = td1.abs(), td2.abs()
            huber1 = torch.where(abs_td1 <= HUBER_KAPPA,
                                 0.5 * td1.pow(2),
                                 HUBER_KAPPA * (abs_td1 - 0.5 * HUBER_KAPPA))
            huber2 = torch.where(abs_td2 <= HUBER_KAPPA,
                                 0.5 * td2.pow(2),
                                 HUBER_KAPPA * (abs_td2 - 0.5 * HUBER_KAPPA))
            w1 = (taus.unsqueeze(0) - (td1 < 0).float()).abs()
            w2 = (taus.unsqueeze(0) - (td2 < 0).float()).abs()
            c_loss = (w * (w1 * huber1).mean(-1, keepdim=True)).mean() \
                   + (w * (w2 * huber2).mean(-1, keepdim=True)).mean()
            # Priority signal: mean absolute quantile TD across both critics
            td_priority = ((abs_td1.mean(-1) + abs_td2.mean(-1)) / 2)
            td_for_log = td_priority.mean().item()
        else:
            q1, q2 = self.critic(s, a)
            t1, t2 = q1 - y, q2 - y
            c_loss = (w * t1.pow(2)).mean() + (w * t2.pow(2)).mean()
            td_priority = ((t1.abs() + t2.abs()) / 2).squeeze(-1)
            td_for_log = td_priority.mean().item()

        self.copt.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.copt.step()

        # ── Actor update ──
        an2, ln2, _ = self.actor.sample(s)
        if self.cfg.use_distributional:
            q1m, q2m = self.critic.q_mean(s, an2)
            a_loss = (al * ln2 - torch.min(q1m, q2m)).mean()
        else:
            q1b, q2b = self.critic(s, an2)
            a_loss = (al * ln2 - torch.min(q1b, q2b)).mean()

        self.aopt.zero_grad()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.aopt.step()

        # ── Alpha update ──
        alpha_loss = -(self.log_alpha * (ln2.detach() + self.target_ent)).mean()
        self.lopt.zero_grad()
        alpha_loss.backward()
        self.lopt.step()

        # ── Soft target update ──
        for tp, p in zip(self.critic_t.parameters(), self.critic.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        # ── Update PER priorities ──
        if self.cfg.use_per and idxs is not None:
            td_np = np.clip(td_priority.detach().cpu().numpy().flatten(), 0, 100.0)
            self.buffer.update_priorities(idxs, td_np)

        return {"c_loss": c_loss.item(), "a_loss": a_loss.item(),
                "alpha": al.item(), "alpha_loss": alpha_loss.item(),
                "td_mean": td_for_log}

    def save(self, path, step):
        torch.save({"step": step,
                     "actor": self.actor.state_dict(),
                     "critic": self.critic.state_dict(),
                     "critic_t": self.critic_t.state_dict(),
                     "la": self.log_alpha.data,
                     "aopt": self.aopt.state_dict(),
                     "copt": self.copt.state_dict(),
                     "lopt": self.lopt.state_dict()}, path)


# ── Utilities ──

def norm_obs(obs, lo, hi):
    return 2.0 * (obs - lo) / (hi - lo + 1e-8) - 1.0


def snapshot_bus_data(env):
    """Extract per-bus voltage, active power, and reactive power from the simulator.
    Returns None if the power flow has diverged (voltage out of physical range)."""
    sim = env.unwrapped.simulator
    bus_snap = {}
    for bus_id, bus in sim.buses.items():
        v_complex = bus.v
        v_mag = float(abs(v_complex))
        # Detect power flow divergence (NR solver didn't converge)
        if v_mag > 2.0 or v_mag < 0.3 or not np.isfinite(v_mag):
            return None
        bus_snap[bus_id] = {
            'v_mag': v_mag,                                        # pu
            'v_ang': float(np.angle(v_complex, deg=True)),         # degrees
            'p': float(bus.p * sim.baseMVA),                       # MW
            'q': float(bus.q * sim.baseMVA),                       # MVAr
        }
    return bus_snap


def evaluate_agent(agent, env, n_rollouts=N_EVAL, horizon=EVAL_HORIZON):
    lo, hi = env.observation_space.low, env.observation_space.high
    returns, q_biases = [], []
    # Accumulate bus data across all steps of all evaluation rollouts
    bus_accum = {}  # bus_id -> {v_mag: [..], p: [..], q: [..]}
    for ep in range(n_rollouts):
        obs, _ = env.reset(seed=ep * 17)
        s0 = norm_obs(obs, lo, hi)
        a0 = agent.act(s0, det=True)
        with torch.no_grad():
            s_t = torch.FloatTensor(s0).unsqueeze(0).to(DEVICE)
            a_t = torch.FloatTensor(a0).unsqueeze(0).to(DEVICE)
            if agent.cfg.use_distributional:
                q1m, q2m = agent.critic.q_mean(s_t, a_t)
                q_pred = torch.min(q1m, q2m).item()
            else:
                q1p, q2p = agent.critic(s_t, a_t)
                q_pred = torch.min(q1p, q2p).item()
        G, g = 0.0, 1.0
        for _ in range(horizon):
            try:
                obs, r, term, trunc, _ = env.step(
                    agent.act(norm_obs(obs, lo, hi), det=True))
            except RuntimeError:
                r, term = -R_CLIP, True
            G += g * r
            g *= GAMMA
            # Capture bus snapshot each eval step (skip if power flow diverged)
            snap = snapshot_bus_data(env)
            if snap is not None:
                for bid, bdata in snap.items():
                    if bid not in bus_accum:
                        bus_accum[bid] = {'v_mag': [], 'v_ang': [], 'p': [], 'q': []}
                    for k in ('v_mag', 'v_ang', 'p', 'q'):
                        bus_accum[bid][k].append(bdata[k])
            if term or trunc:
                break
        returns.append(G)
        q_biases.append(q_pred - G)
    # Average bus data over the evaluation
    bus_avg = {}
    for bid, acc in bus_accum.items():
        bus_avg[bid] = {k: float(np.mean(v)) for k, v in acc.items()}
    return float(np.mean(returns)), float(np.std(returns)), \
           float(np.mean(q_biases)), returns, bus_avg


# ── Training loop ──

def train_one_seed(cfg, seed, results_dir="dsac_results"):
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"{cfg.name}_seed{seed}.pkl")
    bus_path = os.path.join(results_dir, f"bus_data_{cfg.name}_seed{seed}_dsac.pkl")
    if os.path.exists(log_path):
        with open(log_path, "rb") as f:
            print(f"  [cache] loaded {log_path}")
            return pickle.load(f)

    print(f"  [train] {cfg.name}  seed={seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    agent = DSACAgent(cfg)
    log = {"steps": [], "returns": [], "returns_std": [], "alpha": [],
           "q_bias": [], "td_mean": [],
           "priority_gini": [], "priority_ess": [], "priority_entropy": [],
           "all_returns": [], "bus_data": []}

    # Granular bus timeseries: snapshot from training env every BUS_LOG_EVERY steps
    # Each entry: {"step": int, "buses": {bus_id: {v_mag, v_ang, p, q}}}
    bus_timeseries = []

    # Per-episode reward tracking
    ep_rewards_log = []   # list of {"step": int, "ep_return": float, "ep_length": int}
    ep_return = 0.0       # accumulating return for current episode

    tenv = gym.make("ANM6Easy-v0")
    eenv = gym.make("ANM6Easy-v0")
    lo, hi = tenv.observation_space.low, tenv.observation_space.high
    obs, _ = tenv.reset(seed=seed)

    step = 0
    ep_step = 0
    crashes = 0
    next_eval = EVAL_EVERY
    t0 = time.time()

    # Weights directory
    weights_dir = os.path.join(results_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    while step < cfg.total_steps:
        s_norm = norm_obs(obs, lo, hi)
        action = (tenv.action_space.sample() if step < cfg.warmup_steps
                  else agent.act(s_norm))
        try:
            nobs, rew, term, trunc, _ = tenv.step(action)
        except RuntimeError:
            crashes += 1
            if ep_step > 0:
                ep_rewards_log.append({"step": step, "ep_return": ep_return, "ep_length": ep_step})
            ep_return = 0.0
            obs, _ = tenv.reset(seed=seed + step)
            ep_step = 0
            step += 1
            continue

        ep_step += 1
        done = term or trunc or (ep_step >= EPISODE_LEN)
        rew = float(np.clip(rew, -R_CLIP, R_CLIP))
        ep_return += rew
        agent.push(s_norm, action, rew, norm_obs(nobs, lo, hi), float(term))
        obs = nobs
        step += 1

        # ── Granular bus data logging from training env ──
        if step % BUS_LOG_EVERY == 0:
            snap = snapshot_bus_data(tenv)
            if snap is not None:
                bus_timeseries.append({
                    "step": step,
                    "buses": snap,
                })

        if done:
            ep_rewards_log.append({"step": step, "ep_return": ep_return, "ep_length": ep_step})
            ep_return = 0.0
            obs, _ = tenv.reset(seed=seed + step)
            ep_step = 0

        if step >= cfg.warmup_steps and step % cfg.update_every == 0:
            agent.update(step)

        if step > 0 and step % 2_000 == 0 and step % EVAL_EVERY != 0:
            elapsed = time.time() - t0
            eta = elapsed / step * (cfg.total_steps - step)
            pct = 100 * step / cfg.total_steps
            recent_r = ep_rewards_log[-1]["ep_return"] if ep_rewards_log else 0.0
            print(f"      [hb] {cfg.name} seed={seed} | step {step:>7,}/{cfg.total_steps:,} ({pct:.0f}%) | buf={agent.buf_len():,} | last_ep_R={recent_r:.1f} | ETA {int(eta)//60}m{int(eta)%60:02d}s")

        if step >= next_eval:
            mr, sr, qb, per_ep, bus_avg = evaluate_agent(agent, eenv)
            pstats = agent._raw_buf.get_priority_stats()
            p_gini = pstats["gini"] if pstats else 0.0
            p_ess = pstats["ess_norm"] if pstats else 1.0
            p_ent = pstats["entropy_norm"] if pstats else 1.0

            log["steps"].append(step)
            log["returns"].append(mr)
            log["returns_std"].append(sr)
            log["alpha"].append(agent.alpha.item())
            log["q_bias"].append(qb)
            log["td_mean"].append(0.0)
            log["priority_gini"].append(p_gini)
            log["priority_ess"].append(p_ess)
            log["priority_entropy"].append(p_ent)
            log["all_returns"].append(per_ep)
            log["bus_data"].append(bus_avg)

            elapsed = time.time() - t0
            eta = elapsed / step * (cfg.total_steps - step)
            mpc = " > MPC!" if mr > MPC_BASELINE else ""
            print(f"    step {step:>7,} | J={mr:>8.1f}+-{sr:.1f}"
                  f" | a={agent.alpha.item():.4f} | Qbias={qb:>6.1f}"
                  f" | Gini={p_gini:.3f} | ETA {int(eta)//60}m{mpc}")

            with open(log_path, "wb") as f:
                pickle.dump(log, f)
            # Save granular bus timeseries to separate _dsac file
            with open(bus_path, "wb") as f:
                pickle.dump({"experiment": cfg.name, "seed": seed,
                              "bus_log_every": BUS_LOG_EVERY,
                              "timeseries": bus_timeseries}, f)
            # Save episode rewards to separate file
            reward_path = os.path.join(results_dir, f"rewards_{cfg.name}_seed{seed}_dsac.pkl")
            with open(reward_path, "wb") as f:
                pickle.dump({"experiment": cfg.name, "seed": seed,
                              "episodes": ep_rewards_log}, f)
            # Save model weights checkpoint
            ckpt_path = os.path.join(weights_dir, f"{cfg.name}_seed{seed}_step{step}.pt")
            agent.save(ckpt_path, step)
            next_eval = step + EVAL_EVERY

    # ── Final saves ──
    with open(bus_path, "wb") as f:
        pickle.dump({"experiment": cfg.name, "seed": seed,
                      "bus_log_every": BUS_LOG_EVERY,
                      "timeseries": bus_timeseries}, f)
    reward_path = os.path.join(results_dir, f"rewards_{cfg.name}_seed{seed}_dsac.pkl")
    with open(reward_path, "wb") as f:
        pickle.dump({"experiment": cfg.name, "seed": seed,
                      "episodes": ep_rewards_log}, f)
    final_ckpt = os.path.join(weights_dir, f"{cfg.name}_seed{seed}_final.pt")
    agent.save(final_ckpt, step)
    print(f"  Bus data saved: {bus_path} ({len(bus_timeseries)} snapshots)")
    print(f"  Rewards saved: {reward_path} ({len(ep_rewards_log)} episodes)")
    print(f"  Weights saved: {final_ckpt}")

    tenv.close()
    eenv.close()
    elapsed = time.time() - t0
    print(f"  Done {cfg.name} seed={seed} in {elapsed/60:.1f}min crashes={crashes}")
    return log


print("All classes defined. Ready to train.")


# ── Parallel runner ──

import multiprocessing
import concurrent.futures

def _train_worker(args):
    name, seed, results_dir = args
    cfg = EXPERIMENTS[name]
    return name, train_one_seed(cfg, seed, results_dir)

def run_all_experiments(exp_names=None, results_dir="dsac_results", max_workers=4):
    if exp_names is None:
        exp_names = list(EXPERIMENTS.keys())
    tasks = []
    for name in exp_names:
        cfg = EXPERIMENTS[name]
        for seed in cfg.seeds:
            tasks.append((name, seed, results_dir))
    all_logs = {n: [] for n in exp_names}
    total = len(tasks)
    print(f"Running {len(exp_names)} experiments x seeds = {total} total runs.")
    print(f"Parallel: {max_workers} workers\n")
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_train_worker, t) for t in tasks]
        done = 0
        for future in concurrent.futures.as_completed(futures):
            done += 1
            try:
                name, log = future.result()
                all_logs[name].append(log)
                print(f"[{done}/{total}] Finished {name}")
            except Exception as e:
                print(f"Task failed: {e}")
    return all_logs


def load_all_results(exp_names=None, results_dir="dsac_results"):
    if exp_names is None:
        exp_names = list(EXPERIMENTS.keys())
    all_logs = {}
    for name in exp_names:
        cfg = EXPERIMENTS[name]
        logs = []
        for seed in cfg.seeds:
            p = os.path.join(results_dir, f"{name}_seed{seed}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    logs.append(pickle.load(f))
        if logs:
            all_logs[name] = logs
    return all_logs


def aggregate(logs_list):
    if not logs_list:
        return None
    steps = np.array(logs_list[0]["steps"])
    n_seeds = len(logs_list)
    def ci(key):
        mat = np.array([l[key] for l in logs_list])
        m = mat.mean(0)
        if n_seeds == 1:
            return m, m - 1, m + 1
        se = mat.std(0, ddof=1) / math.sqrt(n_seeds)
        t = stats.t.ppf(0.975, df=n_seeds - 1)
        return m, m - t * se, m + t * se
    mean_r, lo_r, hi_r = ci("returns")
    return dict(
        steps=steps, mean_r=mean_r, lo_r=lo_r, hi_r=hi_r,
        mean_alpha=np.array([l["alpha"] for l in logs_list]).mean(0),
        mean_qbias=np.array([l["q_bias"] for l in logs_list]).mean(0),
        mean_gini=np.array([l["priority_gini"] for l in logs_list]).mean(0),
        mean_ess=np.array([l["priority_ess"] for l in logs_list]).mean(0),
        mean_ent=np.array([l["priority_entropy"] for l in logs_list]).mean(0),
        n_seeds=n_seeds,
    )


# ── Main ──

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    RESULTS_DIR = "dsac_results"
    all_logs = run_all_experiments(results_dir=RESULTS_DIR, max_workers=18)

    print("\n=== Post-processing ===")
    agg = {name: aggregate(logs) for name, logs in all_logs.items() if logs}

    COLORS = {
        "SAC_base": "#E24B4A", "SACPER_base": "#185FA5",
        "DSAC_uniform": "#1D9E75", "DSAC_PER": "#EF9F27",
        "SAC_LAP": "#533FAD", "DSAC_LAP": "#A8428C",
    }
    LABELS = {
        "SAC_base": "SAC (uniform)", "SACPER_base": "SAC + PER",
        "DSAC_uniform": "DSAC (uniform)", "DSAC_PER": "DSAC + PER",
        "SAC_LAP": "SAC + LAP", "DSAC_LAP": "DSAC + LAP",
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Plot 1: Main comparison
    fig, ax = plt.subplots(figsize=(13, 5))
    for name in EXPERIMENTS:
        if name not in agg or agg[name] is None:
            continue
        a = agg[name]
        xs = a["steps"] / 1000
        m = np.clip(a["mean_r"], -300, 0)
        lo = np.clip(a["lo_r"], -300, 0)
        hi = np.clip(a["hi_r"], -300, 0)
        c = COLORS.get(name, "gray")
        ax.plot(xs, m, color=c, lw=2, label=LABELS.get(name, name))
        ax.fill_between(xs, lo, hi, color=c, alpha=0.15)
    ax.axhline(MPC_BASELINE, color="black", lw=1, ls=":", alpha=0.6)
    ax.set_xlabel("Training steps (x1000)")
    ax.set_ylabel("Discounted return")
    ax.set_title("DSAC-PER Ablation: Distributional Critics + Prioritized Replay")
    ax.legend(fontsize=9, ncol=2, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=-305)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/fig_main_comparison.pdf", bbox_inches="tight")
    plt.savefig(f"{RESULTS_DIR}/fig_main_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Distributional vs Scalar
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (title, names) in zip(axes, [
        ("Scalar critics", ["SAC_base", "SACPER_base", "SAC_LAP"]),
        ("Quantile critics", ["DSAC_uniform", "DSAC_PER", "DSAC_LAP"]),
    ]):
        for name in names:
            if name not in agg or agg[name] is None:
                continue
            a = agg[name]
            m = np.clip(a["mean_r"], -300, 0)
            lo = np.clip(a["lo_r"], -300, 0)
            hi = np.clip(a["hi_r"], -300, 0)
            c = COLORS.get(name, "gray")
            ax.plot(a["steps"]/1000, m, color=c, lw=2, label=LABELS.get(name, name))
            ax.fill_between(a["steps"]/1000, lo, hi, color=c, alpha=0.15)
        ax.axhline(MPC_BASELINE, lw=1, ls=":", color="gray")
        ax.set_title(title)
        ax.set_xlabel("Steps (x1k)")
        ax.set_ylabel("Return")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=-305)
    plt.suptitle("Scalar vs Distributional Critics")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/fig_scalar_vs_dist.pdf", bbox_inches="tight")
    plt.close()

    # Plot 3: Q-bias comparison
    fig, ax = plt.subplots(figsize=(13, 5))
    for name in EXPERIMENTS:
        if name not in agg or agg[name] is None:
            continue
        a = agg[name]
        ax.plot(a["steps"]/1000, a["mean_qbias"], color=COLORS.get(name, "gray"),
                lw=2, label=LABELS.get(name, name))
    ax.axhline(0, lw=1, color="black", alpha=0.5)
    ax.set_title("Q-value overestimation bias")
    ax.set_xlabel("Steps (x1k)")
    ax.set_ylabel("Q_pred - G_actual")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/fig_q_bias.pdf", bbox_inches="tight")
    plt.close()

    # Plot 4: Priority distribution (PER/LAP variants only)
    per_names = [n for n in ["SACPER_base","DSAC_PER","SAC_LAP","DSAC_LAP"]
                 if n in agg and agg[n] is not None]
    if per_names:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for name in per_names:
            a = agg[name]
            c, lab = COLORS.get(name, "gray"), LABELS.get(name, name)
            axes[0].plot(a["steps"]/1000, a["mean_gini"], color=c, lw=2, label=lab)
            axes[1].plot(a["steps"]/1000, a["mean_ess"], color=c, lw=2, label=lab)
            axes[2].plot(a["steps"]/1000, a["mean_ent"], color=c, lw=2, label=lab)
        for ax, (t, yl) in zip(axes, [
            ("Priority Gini", "Gini"),
            ("Normalized ESS", "ESS/N"),
            ("Priority Entropy", "H/H_max"),
        ]):
            ax.set_title(t); ax.set_ylabel(yl); ax.set_xlabel("Steps (x1k)")
            ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        plt.suptitle("Priority Distribution Evolution")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/fig_priority_dist.pdf", bbox_inches="tight")
        plt.close()

    # ── Plot 5: Bus Data Graphs ──
    # Collect which experiments have bus data
    exps_with_bus = []
    for name in EXPERIMENTS:
        if name not in all_logs or not all_logs[name]:
            continue
        if any(l.get("bus_data") for l in all_logs[name]):
            exps_with_bus.append(name)

    if exps_with_bus:
        # Determine which buses exist from first available data
        sample_log = None
        for name in exps_with_bus:
            for l in all_logs[name]:
                if l.get("bus_data") and len(l["bus_data"]) > 0:
                    sample_log = l
                    break
            if sample_log:
                break
        bus_ids = sorted(sample_log["bus_data"][0].keys()) if sample_log else []
        n_buses = len(bus_ids)

        # Bus type labels
        BUS_LABELS = {
            0: "Bus 0 (Slack/Grid)",
            1: "Bus 1 (Junction)",
            2: "Bus 2 (Junction)",
            3: "Bus 3 (Load+Solar)",
            4: "Bus 4 (Load+Wind)",
            5: "Bus 5 (Load+Storage)",
        }

        # --- Per‑experiment bus voltage profiles ---
        for exp_name in exps_with_bus:
            logs = [l for l in all_logs[exp_name] if l.get("bus_data") and len(l["bus_data"]) > 0]
            if not logs:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=120)
            axes = axes.flatten()
            bus_colors = ['#E24B4A', '#185FA5', '#1D9E75', '#EF9F27', '#533FAD', '#A8428C']

            for idx, bid in enumerate(bus_ids):
                ax = axes[idx]
                for seed_i, log in enumerate(logs):
                    steps = np.array(log["steps"]) / 1000
                    v_mag = [bd[bid]['v_mag'] for bd in log["bus_data"]]
                    ax.plot(steps, v_mag, color=bus_colors[idx],
                            alpha=0.3 + 0.4 * (seed_i == 0), lw=1.5,
                            label=f'Seed {seed_i}' if len(logs) > 1 else None)
                # Mean across seeds
                if len(logs) > 1:
                    min_len = min(len(l["bus_data"]) for l in logs)
                    mean_v = np.mean([[l["bus_data"][t][bid]['v_mag'] for t in range(min_len)] for l in logs], axis=0)
                    steps_common = np.array(logs[0]["steps"][:min_len]) / 1000
                    ax.plot(steps_common, mean_v, color=bus_colors[idx], lw=2.5, label='Mean')
                ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.5)
                ax.axhline(0.9, color='red', ls='--', lw=1, alpha=0.4, label='V_min=0.9')
                ax.axhline(1.1, color='red', ls='--', lw=1, alpha=0.4, label='V_max=1.1')
                ax.set_title(BUS_LABELS.get(bid, f'Bus {bid}'), fontsize=11, fontweight='bold')
                ax.set_xlabel('Steps (×1000)')
                ax.set_ylabel('Voltage (pu)')
                ax.grid(alpha=0.3)
                if idx == 0:
                    ax.legend(fontsize=8)

            plt.suptitle(f'Bus Voltage Magnitude — {LABELS.get(exp_name, exp_name)}',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/fig_bus_voltage_{exp_name}.png", dpi=150, bbox_inches='tight')
            plt.savefig(f"{RESULTS_DIR}/fig_bus_voltage_{exp_name}.pdf", bbox_inches='tight')
            plt.close()

        # --- Per‑experiment bus active power profiles ---
        for exp_name in exps_with_bus:
            logs = [l for l in all_logs[exp_name] if l.get("bus_data") and len(l["bus_data"]) > 0]
            if not logs:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=120)
            axes = axes.flatten()

            for idx, bid in enumerate(bus_ids):
                ax = axes[idx]
                for seed_i, log in enumerate(logs):
                    steps = np.array(log["steps"]) / 1000
                    p_val = [bd[bid]['p'] for bd in log["bus_data"]]
                    ax.plot(steps, p_val, color=bus_colors[idx],
                            alpha=0.3 + 0.4 * (seed_i == 0), lw=1.5,
                            label=f'Seed {seed_i}' if len(logs) > 1 else None)
                if len(logs) > 1:
                    min_len = min(len(l["bus_data"]) for l in logs)
                    mean_p = np.mean([[l["bus_data"][t][bid]['p'] for t in range(min_len)] for l in logs], axis=0)
                    steps_common = np.array(logs[0]["steps"][:min_len]) / 1000
                    ax.plot(steps_common, mean_p, color=bus_colors[idx], lw=2.5, label='Mean')
                ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)
                ax.set_title(BUS_LABELS.get(bid, f'Bus {bid}'), fontsize=11, fontweight='bold')
                ax.set_xlabel('Steps (×1000)')
                ax.set_ylabel('Active Power (MW)')
                ax.grid(alpha=0.3)
                if idx == 0:
                    ax.legend(fontsize=8)

            plt.suptitle(f'Bus Active Power — {LABELS.get(exp_name, exp_name)}',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/fig_bus_active_power_{exp_name}.png", dpi=150, bbox_inches='tight')
            plt.savefig(f"{RESULTS_DIR}/fig_bus_active_power_{exp_name}.pdf", bbox_inches='tight')
            plt.close()

        # --- Per‑experiment bus reactive power profiles ---
        for exp_name in exps_with_bus:
            logs = [l for l in all_logs[exp_name] if l.get("bus_data") and len(l["bus_data"]) > 0]
            if not logs:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=120)
            axes = axes.flatten()

            for idx, bid in enumerate(bus_ids):
                ax = axes[idx]
                for seed_i, log in enumerate(logs):
                    steps = np.array(log["steps"]) / 1000
                    q_val = [bd[bid]['q'] for bd in log["bus_data"]]
                    ax.plot(steps, q_val, color=bus_colors[idx],
                            alpha=0.3 + 0.4 * (seed_i == 0), lw=1.5,
                            label=f'Seed {seed_i}' if len(logs) > 1 else None)
                if len(logs) > 1:
                    min_len = min(len(l["bus_data"]) for l in logs)
                    mean_q = np.mean([[l["bus_data"][t][bid]['q'] for t in range(min_len)] for l in logs], axis=0)
                    steps_common = np.array(logs[0]["steps"][:min_len]) / 1000
                    ax.plot(steps_common, mean_q, color=bus_colors[idx], lw=2.5, label='Mean')
                ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)
                ax.set_title(BUS_LABELS.get(bid, f'Bus {bid}'), fontsize=11, fontweight='bold')
                ax.set_xlabel('Steps (×1000)')
                ax.set_ylabel('Reactive Power (MVAr)')
                ax.grid(alpha=0.3)
                if idx == 0:
                    ax.legend(fontsize=8)

            plt.suptitle(f'Bus Reactive Power — {LABELS.get(exp_name, exp_name)}',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/fig_bus_reactive_power_{exp_name}.png", dpi=150, bbox_inches='tight')
            plt.savefig(f"{RESULTS_DIR}/fig_bus_reactive_power_{exp_name}.pdf", bbox_inches='tight')
            plt.close()

        # --- Combined comparison: all experiments, one plot per bus quantity ---
        for quantity, q_key, ylabel, ylbl in [
            ('Voltage Magnitude', 'v_mag', 'Voltage (pu)', 'voltage'),
            ('Active Power', 'p', 'Active Power (MW)', 'active_power'),
            ('Reactive Power', 'q', 'Reactive Power (MVAr)', 'reactive_power'),
        ]:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=120)
            axes = axes.flatten()
            for idx, bid in enumerate(bus_ids):
                ax = axes[idx]
                for exp_name in exps_with_bus:
                    logs = [l for l in all_logs[exp_name]
                            if l.get("bus_data") and len(l["bus_data"]) > 0]
                    if not logs:
                        continue
                    min_len = min(len(l["bus_data"]) for l in logs)
                    if min_len == 0:
                        continue
                    mean_vals = np.mean(
                        [[l["bus_data"][t][bid][q_key] for t in range(min_len)] for l in logs],
                        axis=0
                    )
                    steps = np.array(logs[0]["steps"][:min_len]) / 1000
                    ax.plot(steps, mean_vals, color=COLORS.get(exp_name, 'gray'),
                            lw=2, label=LABELS.get(exp_name, exp_name))
                if q_key == 'v_mag':
                    ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.5)
                    ax.axhline(0.9, color='red', ls='--', lw=0.8, alpha=0.3)
                    ax.axhline(1.1, color='red', ls='--', lw=0.8, alpha=0.3)
                else:
                    ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)
                ax.set_title(BUS_LABELS.get(bid, f'Bus {bid}'), fontsize=11, fontweight='bold')
                ax.set_xlabel('Steps (×1000)')
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
                if idx == 0:
                    ax.legend(fontsize=7, ncol=2)
            plt.suptitle(f'Bus {quantity} — All Experiments Compared',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/fig_bus_{ylbl}_comparison.png", dpi=150, bbox_inches='tight')
            plt.savefig(f"{RESULTS_DIR}/fig_bus_{ylbl}_comparison.pdf", bbox_inches='tight')
            plt.close()

        print(f"Bus data plots saved to {RESULTS_DIR}/")
    else:
        print("No bus data available for plotting (run training first).")

    # ── Plot 6: Training Episode Rewards (from _dsac reward files) ──
    print("\n── Generating training reward plots ──")
    import glob as _glob

    def smooth(arr, window=20):
        """Simple moving average for noisy episode rewards."""
        if len(arr) < window:
            return arr
        return np.convolve(arr, np.ones(window)/window, mode='valid')

    # Load reward files
    reward_data = {}  # name -> [{steps, returns}, ...]
    for name in EXPERIMENTS:
        cfg = EXPERIMENTS[name]
        seeds_data = []
        for seed in cfg.seeds:
            rp = os.path.join(RESULTS_DIR, f"rewards_{name}_seed{seed}_dsac.pkl")
            if os.path.exists(rp):
                with open(rp, "rb") as f:
                    rd = pickle.load(f)
                    eps = rd.get("episodes", [])
                    if eps:
                        seeds_data.append(eps)
        if seeds_data:
            reward_data[name] = seeds_data

    if reward_data:
        # Plot 6a: Per-experiment training rewards (all seeds + smoothed mean)
        for exp_name, seeds_eps in reward_data.items():
            fig, ax = plt.subplots(figsize=(13, 5))
            for seed_i, eps in enumerate(seeds_eps):
                steps = [e["step"] for e in eps]
                rets = [e["ep_return"] for e in eps]
                ax.scatter(np.array(steps)/1000, rets, alpha=0.08, s=3,
                           color=COLORS.get(exp_name, "gray"), label=None)
                sm = smooth(np.array(rets))
                sm_steps = np.array(steps[:len(sm)])/1000
                ax.plot(sm_steps, sm, color=COLORS.get(exp_name, "gray"),
                        lw=2, alpha=0.5 + 0.3*(seed_i==0),
                        label=f"Seed {seed_i} (smoothed)")
            ax.axhline(MPC_BASELINE, color="black", lw=1, ls=":", alpha=0.6,
                        label=f"MPC baseline ({MPC_BASELINE})")
            ax.set_xlabel("Training steps (×1000)", fontsize=11)
            ax.set_ylabel("Episode return", fontsize=11)
            ax.set_title(f"Training Episode Returns — {LABELS.get(exp_name, exp_name)}",
                         fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/fig_train_rewards_{exp_name}.png", dpi=150, bbox_inches="tight")
            plt.savefig(f"{RESULTS_DIR}/fig_train_rewards_{exp_name}.pdf", bbox_inches="tight")
            plt.close()

        # Plot 6b: All experiments compared (smoothed training rewards)
        fig, ax = plt.subplots(figsize=(13, 5))
        for exp_name, seeds_eps in reward_data.items():
            # Average across seeds: interpolate to common step grid
            all_rets_smooth = []
            for eps in seeds_eps:
                rets = [e["ep_return"] for e in eps]
                all_rets_smooth.append(smooth(np.array(rets), window=30))
            min_len = min(len(s) for s in all_rets_smooth)
            if min_len > 0:
                mean_sm = np.mean([s[:min_len] for s in all_rets_smooth], axis=0)
                steps_ref = [e["step"] for e in seeds_eps[0]][:min_len]
                ax.plot(np.array(steps_ref)/1000, mean_sm,
                        color=COLORS.get(exp_name, "gray"), lw=2,
                        label=LABELS.get(exp_name, exp_name))
        ax.axhline(MPC_BASELINE, color="black", lw=1, ls=":", alpha=0.6)
        ax.set_xlabel("Training steps (×1000)", fontsize=11)
        ax.set_ylabel("Episode return (smoothed)", fontsize=11)
        ax.set_title("Training Rewards — All Experiments Compared",
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, ncol=2, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/fig_train_rewards_comparison.png", dpi=150, bbox_inches="tight")
        plt.savefig(f"{RESULTS_DIR}/fig_train_rewards_comparison.pdf", bbox_inches="tight")
        plt.close()

        print(f"  Training reward plots saved to {RESULTS_DIR}/")
    else:
        print("  No reward data files found.")

    # ── Plot 7: Alpha (entropy temperature) decay ──
    fig, ax = plt.subplots(figsize=(13, 4))
    for name in EXPERIMENTS:
        if name not in agg or agg[name] is None:
            continue
        a = agg[name]
        ax.plot(a["steps"]/1000, a["mean_alpha"], color=COLORS.get(name, "gray"),
                lw=2, label=LABELS.get(name, name))
    ax.axhline(ALPHA_FLOOR, lw=1, ls=":", color="gray", alpha=0.6)
    ax.text(0, ALPHA_FLOOR + 0.003, f" α floor ({ALPHA_FLOOR})", fontsize=8, color="gray")
    ax.set_xlabel("Steps (×1000)", fontsize=11)
    ax.set_ylabel("α (entropy temperature)", fontsize=11)
    ax.set_title("Entropy Temperature α Decay", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.25)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/fig_alpha_decay.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{RESULTS_DIR}/fig_alpha_decay.pdf", bbox_inches="tight")
    plt.close()

    # Summary table
    print("\n" + "="*90)
    print(f"{'Experiment':18s} {'FinalReturn':>12} {'BestReturn':>12} {'AvgQbias':>10} {'AvgGini':>9}")
    print("-"*90)
    for name in EXPERIMENTS:
        if name not in all_logs or not all_logs[name]:
            continue
        logs = all_logs[name]
        finals = [l["returns"][-1] for l in logs if l.get("returns")]
        bests = [max(l["returns"]) for l in logs if l.get("returns")]
        qb = np.mean([np.mean(l["q_bias"]) for l in logs if l.get("q_bias")])
        gi = np.mean([np.mean(l["priority_gini"]) for l in logs if l.get("priority_gini")])
        print(f"{name:18s} {np.mean(finals):>12.2f} {np.mean(bests):>12.2f} {qb:>10.2f} {gi:>9.4f}")
    print("="*90)

    with open(f"{RESULTS_DIR}/summary.txt", "w") as f:
        for name in EXPERIMENTS:
            if name not in all_logs or not all_logs[name]:
                continue
            logs = all_logs[name]
            finals = [l["returns"][-1] for l in logs if l.get("returns")]
            f.write(f"{name}: final={np.mean(finals):.2f} seeds={len(finals)}\n")

    print(f"\nAll plots saved to {RESULTS_DIR}/")
    print("Done!")
