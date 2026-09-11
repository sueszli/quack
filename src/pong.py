from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

import src.utils  # noqa: F401
from src.utils import data_path

H = W = 64
FRAMES = 4
PADDLE_X = 0.06
PADDLE_HW = 0.015
BALL_R = 0.02
PADDLE_SPEED = 0.03
BALL_SPEED = 0.02
SPEED_RAMP = 1.12
MAX_SPEED_MULT = 4.0
SPIN = 0.6
PARAM_BOUNDS = {"ball_speed_mult": (0.5, 2.0), "paddle_half_h": (0.04, 0.16)}
PARAM_DEFAULTS = {"ball_speed_mult": 1.0, "paddle_half_h": 0.08}


def grid(device: torch.device) -> tuple[Tensor, Tensor]:
    ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return yy[None], xx[None]


def as_color(n: int, color, device: torch.device) -> Tensor:
    c = color if torch.is_tensor(color) else torch.tensor(color, device=device, dtype=torch.float32)
    c = c[None].expand(n, -1) if c.dim() == 1 else c
    assert c.shape == (n, 3), c.shape
    return c[:, :, None, None]


def canvas(n: int, device: torch.device, color=(0.0, 0.0, 0.0)) -> Tensor:
    return as_color(n, color, device).expand(n, 3, H, W).clone()


def paint(img: Tensor, mask: Tensor, color) -> Tensor:
    assert mask.shape == (img.shape[0], H, W) and mask.dtype == torch.bool
    return img.copy_(torch.where(mask[:, None], as_color(img.shape[0], color, img.device), img))


def draw_rect(img: Tensor, cx: Tensor, cy: Tensor, hw, hh, color) -> Tensor:
    yy, xx = grid(img.device)
    hw = hw[:, None, None] if torch.is_tensor(hw) else hw
    hh = hh[:, None, None] if torch.is_tensor(hh) else hh
    return paint(img, ((xx - cx[:, None, None]).abs() <= hw) & ((yy - cy[:, None, None]).abs() <= hh), color)


def draw_circle(img: Tensor, cx: Tensor, cy: Tensor, r, color) -> Tensor:
    yy, xx = grid(img.device)
    r = r[:, None, None] if torch.is_tensor(r) else r
    return paint(img, (xx - cx[:, None, None]) ** 2 + (yy - cy[:, None, None]) ** 2 <= r**2, color)


@dataclass
class State:
    bx: Tensor
    by: Tensor
    vx: Tensor
    vy: Tensor
    py: Tensor
    speed: Tensor
    hits: Tensor
    paddle_hh: Tensor


def draw_play(img: Tensor, s: State, ball=(1.0, 1.0, 1.0), paddle=(1.0, 1.0, 1.0)) -> Tensor:
    draw_rect(img, torch.full_like(s.py, PADDLE_X), s.py, PADDLE_HW, s.paddle_hh, paddle)
    return draw_circle(img, s.bx, s.by, BALL_R, ball)


class Clean:
    PARAMS: dict[str, float] = {}

    def render(self, s: State, t: Tensor, rng: torch.Generator) -> Tensor:
        return draw_play(canvas(s.bx.shape[0], s.bx.device), s)


class RandomBg:
    PARAMS: dict[str, float] = {}
    color: Tensor | None = None

    def render(self, s: State, t: Tensor, rng: torch.Generator) -> Tensor:
        n, dev = s.bx.shape[0], s.bx.device
        new = 0.7 * torch.rand(n, 3, device=dev, generator=rng)
        self.color = new if self.color is None else torch.where((t == 0)[:, None], new, self.color)
        return draw_play(canvas(n, dev, self.color), s)


class MovingDistractor:
    PARAMS: dict[str, float] = {}

    def render(self, s: State, t: Tensor, rng: torch.Generator) -> Tensor:
        img = canvas(s.bx.shape[0], s.bx.device)
        tf = t.float()
        draw_circle(img, 0.55 + 0.35 * torch.sin(0.05 * tf), 0.5 + 0.4 * torch.cos(0.031 * tf), BALL_R, (1.0, 0.3, 0.3))
        return draw_play(img, s)


class Flowers:
    PARAMS: dict[str, float] = {}
    K = 12
    pos: Tensor | None = None
    col: Tensor | None = None

    def render(self, s: State, t: Tensor, rng: torch.Generator) -> Tensor:
        n, dev = s.bx.shape[0], s.bx.device
        pos = torch.rand(n, self.K, 2, device=dev, generator=rng)
        col = 0.8 * torch.rand(n, self.K, 3, device=dev, generator=rng)
        m = (t == 0)[:, None, None]
        self.pos = pos if self.pos is None else torch.where(m, pos, self.pos)
        self.col = col if self.col is None else torch.where(m, col, self.col)
        img = canvas(n, dev)
        for k in range(self.K):
            draw_circle(img, self.pos[:, k, 0], self.pos[:, k, 1], 0.03, self.col[:, k])
        return draw_play(img, s)


class FastBall(Clean):
    PARAMS = {"ball_speed_mult": 1.6}


SKINS = {"clean": Clean, "random_bg": RandomBg, "moving_distractor": MovingDistractor, "flowers": Flowers, "fast_ball": FastBall}


def clamp_params(p: dict) -> dict[str, float]:
    return {k: float(min(hi, max(lo, p.get(k, PARAM_DEFAULTS[k])))) for k, (lo, hi) in PARAM_BOUNDS.items()}


class Pong:
    def __init__(self, n: int, skin, device: str = "cuda", max_steps: int = 1000, seed: int = 0):
        self.n, self.device, self.max_steps, self.skin = n, torch.device(device), max_steps, skin
        self.params = clamp_params(skin.PARAMS)
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.s = State(*[torch.zeros(n, device=self.device) for _ in range(8)])
        self.t = torch.zeros(n, device=self.device, dtype=torch.long)
        self.frames = torch.zeros(n, FRAMES, 3, H, W, device=self.device, dtype=torch.uint8)
        self.fi = 0
        self.reset(torch.ones(n, device=self.device, dtype=torch.bool))
        img = self.render()
        for _ in range(FRAMES):
            self.push(img)

    def uniform(self, m: Tensor, lo: float, hi: float) -> Tensor:
        return lo + (hi - lo) * torch.rand(int(m.sum()), device=self.device, generator=self.rng)

    def reset(self, m: Tensor) -> None:
        if not m.any():
            return
        s = self.s
        s.bx[m] = 0.5
        s.by[m] = self.uniform(m, 0.2, 0.8)
        ang = self.uniform(m, -0.7, 0.7)
        s.vx[m] = -BALL_SPEED * torch.cos(ang)
        s.vy[m] = BALL_SPEED * torch.sin(ang)
        s.py[m] = self.uniform(m, 0.3, 0.7)
        s.speed[m] = self.params["ball_speed_mult"]
        s.hits[m] = 0
        s.paddle_hh[m] = self.params["paddle_half_h"]
        self.t[m] = 0

    def render(self) -> Tensor:
        img = self.skin.render(self.s, self.t, self.rng)
        assert img.shape == (self.n, 3, H, W) and img.dtype == torch.float32 and img.device.type == self.device.type, (img.shape, img.dtype, img.device)
        assert torch.isfinite(img).all()
        return img.clamp_(0, 1).mul_(255).to(torch.uint8)

    def push(self, img8: Tensor) -> None:
        self.frames[:, self.fi] = img8
        self.fi = (self.fi + 1) % FRAMES

    def obs(self) -> Tensor:
        idx = (torch.arange(FRAMES, device=self.device) + self.fi) % FRAMES
        return self.frames[:, idx].flatten(1, 2)

    @torch.no_grad()
    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        assert action.shape == (self.n,) and action.dtype == torch.long
        s = self.s
        s.py.add_(torch.where(action == 0, -PADDLE_SPEED, PADDLE_SPEED)).clamp_(s.paddle_hh, 1 - s.paddle_hh)
        s.bx.add_(s.vx * s.speed)
        s.by.add_(s.vy * s.speed)
        s.vy[(s.by < BALL_R) | (s.by > 1 - BALL_R)] *= -1
        s.by.clamp_(BALL_R, 1 - BALL_R)
        wall = s.bx > 1 - BALL_R
        s.vx[wall] = -s.vx[wall].abs()
        s.bx.clamp_(max=1 - BALL_R)
        hit = (s.bx - BALL_R <= PADDLE_X + PADDLE_HW) & (s.vx < 0) & ((s.by - s.py).abs() <= s.paddle_hh + BALL_R)
        s.vx[hit] = s.vx[hit].abs()
        s.vy[hit] += SPIN * BALL_SPEED * (s.by[hit] - s.py[hit]) / s.paddle_hh[hit]
        norm = BALL_SPEED / torch.sqrt(s.vx**2 + s.vy**2)
        s.vx.mul_(norm)
        s.vy.mul_(norm)
        s.bx[hit] = PADDLE_X + PADDLE_HW + BALL_R
        s.speed[hit] = (s.speed[hit] * SPEED_RAMP).clamp(max=MAX_SPEED_MULT)
        s.hits[hit] += 1
        miss = s.bx < 0
        self.t += 1
        done = miss | (self.t >= self.max_steps)
        reward = hit.float() - miss.float()
        info = {"hits": s.hits.clone(), "miss": miss}
        self.reset(done)
        self.push(self.render())
        return self.obs(), reward, done, info


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(FRAMES * 3, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 4 * 4, 256), nn.ReLU())
        self.pi = nn.Linear(256, 2)
        self.v = nn.Linear(256, 1)

    def forward(self, obs8: Tensor) -> tuple[Tensor, Tensor]:
        h = self.net(obs8.float() / 255.0)
        return self.pi(h), self.v(h).squeeze(-1)


def train(skin: str, seed: int, num_envs: int = 2048, total_steps: int = 25_000_000, rollout: int = 64, epochs: int = 3, minibatches: int = 8, lr: float = 5e-4, gamma: float = 0.99, lam: float = 0.95, clip: float = 0.2, ent: float = 0.01, device: str = "cuda") -> Path:
    torch.manual_seed(seed)
    dev = torch.device(device)
    out = data_path("pong", f"{skin}_s{seed}")
    (out / "config.json").write_text(json.dumps({"skin": skin, "seed": seed, "num_envs": num_envs, "total_steps": total_steps}))
    log = (out / "metrics.jsonl").open("a")
    env = Pong(num_envs, SKINS[skin](), device=device, seed=seed)
    agent = Agent().to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    N, T = num_envs, rollout
    obs_buf = torch.zeros(T, N, FRAMES * 3, H, W, dtype=torch.uint8, device=dev)
    act_buf, logp_buf, rew_buf, done_buf, val_buf = torch.zeros(T, N, dtype=torch.long, device=dev), *[torch.zeros(T, N, device=dev) for _ in range(4)]
    obs = env.obs()
    ep_ret, ep_len = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
    fin: list[Tensor] = []
    iters = total_steps // (N * T)
    assert iters > 0
    t0 = time.time()
    for it in range(1, iters + 1):
        for t in range(T):
            with torch.no_grad():
                logits, v = agent(obs)
                dist = Categorical(logits=logits)
                a = dist.sample()
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, a, dist.log_prob(a), v
            obs, r, d, info = env.step(a)
            rew_buf[t], done_buf[t] = r, d.float()
            ep_ret += r
            ep_len += 1
            if d.any():
                fin.append(torch.stack([ep_ret[d], ep_len[d], info["hits"][d].float()], 1))
                ep_ret[d] = 0
                ep_len[d] = 0
        with torch.no_grad():
            _, next_v = agent(obs)
            adv = torch.zeros_like(rew_buf)
            last = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nv = next_v if t == T - 1 else val_buf[t + 1]
                nonterm = 1.0 - done_buf[t]
                last = rew_buf[t] + gamma * nv * nonterm - val_buf[t] + gamma * lam * nonterm * last
                adv[t] = last
            ret = adv + val_buf
        b_obs, b_act, b_logp, b_adv, b_ret = obs_buf.flatten(0, 1), act_buf.flatten(), logp_buf.flatten(), adv.flatten(), ret.flatten()
        B = N * T
        for _ in range(epochs):
            perm = torch.randperm(B, device=dev)
            for i in range(0, B, B // minibatches):
                idx = perm[i : i + B // minibatches]
                logits, v = agent(b_obs[idx])
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(b_act[idx]) - b_logp[idx]).exp()
                a_ = (b_adv[idx] - b_adv[idx].mean()) / (b_adv[idx].std() + 1e-8)
                loss = torch.max(-a_ * ratio, -a_ * ratio.clamp(1 - clip, 1 + clip)).mean() + 0.25 * (v - b_ret[idx]).pow(2).mean() - ent * dist.entropy().mean()
                assert torch.isfinite(loss)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()
        if fin:
            m = torch.cat(fin).mean(0).tolist()
            fin = []
            rec = {"it": it, "steps": it * N * T, "ep_return": m[0], "ep_len": m[1], "hits": m[2], "sps": it * N * T / (time.time() - t0)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"it {it:4d} steps {rec['steps'] / 1e6:6.2f}M ret {m[0]:6.2f} len {m[1]:6.1f} hits {m[2]:5.1f} sps {rec['sps']:8.0f}")
        if it % 20 == 0 or it == iters:
            torch.save(agent.state_dict(), out / "model.pt")
    return out


@torch.no_grad()
def evaluate(run: Path, num_envs: int = 512, episodes: int = 512, seed: int = 1000, random_policy: bool = False, device: str = "cuda") -> list[dict]:
    agent = Agent().to(device)
    if not random_policy:
        agent.load_state_dict(torch.load(run / "model.pt", map_location=device))
    res = []
    for name, skin in SKINS.items():
        env = Pong(num_envs, skin(), device=device, seed=seed)
        obs = env.obs()
        hits, lens, l, n = [], [], torch.zeros(num_envs, device=device), 0
        while n < episodes:
            a = torch.randint(0, 2, (num_envs,), device=device) if random_policy else Categorical(logits=agent(obs)[0]).sample()
            obs, _, d, info = env.step(a)
            l += 1
            if d.any():
                hits.append(info["hits"][d].float())
                lens.append(l[d])
                n += int(d.sum())
                l[d] = 0
        h = torch.cat(hits)
        res.append({"skin": name, "hits_mean": h.mean().item(), "hits_std": h.std().item(), "ep_len": torch.cat(lens).mean().item(), "n": len(h)})
        print(f"{name:20s} hits {res[-1]['hits_mean']:6.2f} ± {res[-1]['hits_std']:5.2f}  len {res[-1]['ep_len']:6.1f}")
    (run / ("eval_random.json" if random_policy else "eval.json")).write_text(json.dumps(res, indent=1))
    return res


def report() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = data_path("pong")
    table: dict[str, dict[str, list[float]]] = {}
    for f in root.glob("*/eval*.json"):
        cond = "random" if f.name == "eval_random.json" else "train:" + json.loads((f.parent / "config.json").read_text())["skin"]
        for r in json.loads(f.read_text()):
            table.setdefault(cond, {}).setdefault(r["skin"], []).append(r["hits_mean"])
    assert table, root
    skins, conds = sorted({s for d in table.values() for s in d}), sorted(table)
    x, w = np.arange(len(skins)), 0.8 / len(conds)
    fig, ax = plt.subplots(figsize=(1.6 * len(skins) + 2, 4))
    for i, c in enumerate(conds):
        vals = [table[c].get(s, [np.nan]) for s in skins]
        ax.bar(x + i * w, [np.mean(v) for v in vals], w, yerr=[np.std(v) for v in vals], capsize=3, label=f"{c} (n={max(map(len, vals))})")
        print(c, {s: round(float(np.mean(v)), 2) for s, v in zip(skins, vals)})
    ax.set_xticks(x + w * (len(conds) - 1) / 2)
    ax.set_xticklabels(skins, rotation=20)
    ax.set_ylabel("hits / episode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "report.png", dpi=120)
    return root / "report.png"


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--skin", default="clean", choices=SKINS)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--total-steps", type=int, default=25_000_000)
    t.add_argument("--num-envs", type=int, default=2048)
    e = sub.add_parser("eval")
    e.add_argument("run", type=Path)
    e.add_argument("--random-policy", action="store_true")
    sub.add_parser("report")
    a = p.parse_args()
    if a.cmd == "train":
        evaluate(train(a.skin, a.seed, num_envs=a.num_envs, total_steps=a.total_steps))
    elif a.cmd == "eval":
        evaluate(a.run, random_policy=a.random_policy)
    else:
        print(report())


if __name__ == "__main__":
    sys.exit(main())
