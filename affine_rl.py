from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Optional, Any, List, Dict

import gymnasium as gym
from gymnasium import spaces
import requests
from requests.exceptions import RequestException

from dotenv import load_dotenv

load_dotenv()


RESET = "\033[0m"
BLUE = "\033[94m"  # agent
GREEN = "\033[92m"  # user
ORANGE = "\033[38;5;208m"  # system (approx orange)


# ---------------------------
# Minimal Chutes HTTP client
# ---------------------------
@dataclass
class ChutesClient:
    """Client wrapper for Chutes with optional streaming."""

    api_key: Optional[str] = None
    model: Optional[str] = "deepseek-ai/DeepSeek-V3-0324"
    BASE_URL: str = os.getenv("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")
    timeout_s: int = 60

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 256, stream: bool = False) -> str:
        url = f"{self.BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        try:
            with requests.post(url, headers=headers, json=payload, timeout=self.timeout_s, stream=stream) as resp:
                resp.raise_for_status()
                if stream:
                    chunks = []
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        try:
                            import json

                            obj = json.loads(data)
                            delta = obj["choices"][0]["delta"].get("content", "")
                            if delta:
                                print(delta, end="", flush=True)
                                chunks.append(delta)
                        except Exception:
                            continue
                    print()  # newline after stream
                    return "".join(chunks).strip()
                else:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
        except RequestException as e:
            return f"(chutes error) {type(e).__name__}: {str(e)[:160]}"


# ----------------------------------------
# Reward functions and utilities (pluggable)
# ----------------------------------------
def default_reward_fn(
    history: List[Dict[str, str]],
    assistant_utterance: str,
    user_reply: str,
    turn_idx: int,
) -> float:
    """Simple baseline reward encouraging concision + engagement."""
    if not assistant_utterance.strip():
        return -0.5
    reward = 0.0
    reward += 1.0 if len(assistant_utterance.split()) <= 60 else 0.2
    reward += 0.5 if len(user_reply.split()) >= 4 else 0.0
    return reward


@dataclass
class ChutesEnvConfig:
    system_prompt: str = "You are a helpful, concise assistant talking to a user. Keep answers brief."
    max_turns: int = 6
    temperature: float = 0.7
    max_tokens: int = 256
    max_char_obs: int = 4096
    max_char_act: int = 1024
    reward_fn: Callable[[List[Dict[str, str]], str, str, int], float] = default_reward_fn


class ChutesConversationEnv(gym.Env):
    """Multi‑turn dialog environment where the *environment* is powered by Chutes."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, client: ChutesClient, config: Optional[ChutesEnvConfig] = None):
        super().__init__()
        self.client = client
        self.cfg = config or ChutesEnvConfig()
        self.observation_space = spaces.Text(max_length=self.cfg.max_char_obs)
        self.action_space = spaces.Text(max_length=self.cfg.max_char_act)
        self._turn = 0
        self._messages: List[Dict[str, str]] = []
        self._last_user_reply: str = ""

    # -------------- Gym API --------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> tuple[str, Dict[str, Any]]:
        super().reset(seed=seed)
        self._turn = 0
        self._messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": options.get("user_seed", "Hello !") if options else "Hello !"},
        ]
        obs = self._serialize_messages()
        return obs, {"messages": list(self._messages)}

    def step(self, action: str) -> tuple[str, float, bool, bool, Dict[str, Any]]:
        assistant_utterance = (action or "").strip()[: self.cfg.max_char_act]
        self._messages.append({"role": "assistant", "content": assistant_utterance})
        user_reply = self.client.chat(self._messages, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens, stream=True)
        self._last_user_reply = user_reply
        self._messages.append({"role": "user", "content": user_reply})
        reward = self.cfg.reward_fn(self._messages, assistant_utterance, user_reply, self._turn)
        self._turn += 1
        terminated = self._turn >= self.cfg.max_turns
        truncated = False
        obs = self._serialize_messages()
        info = {
            "messages": list(self._messages),
            "turn": self._turn,
            "user_reply": user_reply,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        print(self._pretty_transcript())

    # -------------- Helpers --------------
    def _serialize_messages(self) -> str:
        lines = [f"{m['role'].upper()}: {m['content']}" for m in self._messages[-20:]]
        return "\n".join(lines)[-self.cfg.max_char_obs :]

    def _pretty_transcript(self) -> str:
        def tag(r):
            if r == "system":
                return f"{ORANGE}[SYS]{RESET}"
            elif r == "user":
                return f"{GREEN}[USER]{RESET}"
            elif r == "assistant":
                return f"{BLUE}[AGENT]{RESET}"
            return r

        return "\n".join(f"{tag(m['role'])} {m['content']}" for m in self._messages)


# ------------------------------------------------------
# Optional: Wrapper to convert Text action/obs to token IDs
# ------------------------------------------------------
class TextToIDsWrapper(gym.Wrapper):
    def __init__(self, env: ChutesConversationEnv, tokenizer, max_obs_tokens: int = 512, max_act_tokens: int = 128):
        super().__init__(env)
        self.tokenizer = tokenizer
        self.max_obs_tokens = max_obs_tokens
        self.max_act_tokens = max_act_tokens
        self.observation_space = spaces.Box(low=0, high=tokenizer.vocab_size - 1, shape=(max_obs_tokens,), dtype=int)
        self.action_space = spaces.Box(low=0, high=tokenizer.vocab_size - 1, shape=(max_act_tokens,), dtype=int)

    def reset(self, **kwargs):
        text_obs, info = self.env.reset(**kwargs)
        return self._encode_obs(text_obs), info

    def step(self, action_ids):
        action_text = self._decode_act(action_ids)
        text_obs, reward, terminated, truncated, info = self.env.step(action_text)
        return self._encode_obs(text_obs), reward, terminated, truncated, info

    def _encode_obs(self, text: str):
        ids = self.tokenizer.encode(text, add_special_tokens=False)[: self.max_obs_tokens]
        return ids + [0] * (self.max_obs_tokens - len(ids))

    def _decode_act(self, ids):
        return self.tokenizer.decode([int(x) for x in ids], skip_special_tokens=True)


# ------------------------------------------------------
# Assistant policy powered by a Chutes model (agent side)
# ------------------------------------------------------
@dataclass
class ChutesActorPolicy:
    client: ChutesClient
    system_prompt: str = "You are the AGENT. Be helpful and concise."
    max_tokens: int = 256
    temperature: float = 0.7

    def __call__(self, observation_text: str) -> str:
        lines = [ln for ln in observation_text.split("\n") if ln.strip()]
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        for ln in lines:
            if ":" in ln:
                head, content = ln.split(":", 1)
                head = head.strip().lower()
                if head.startswith("user") or "chutes" in head:
                    role = "user"
                elif head.startswith("assist"):
                    role = "assistant"
                elif head.startswith("sys") or head.startswith("system"):
                    role = "system"
                else:
                    role = "user"
                messages.append({"role": role, "content": content.strip()})
        if messages and messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": "(continue)"})

        print("\n[AGENT prompt]")
        for m in messages:
            if m["role"] == "user":
                print(f"{GREEN}USER>{RESET} {m['content']}")
            elif m["role"] == "system":
                print(f"{ORANGE}SYS>{RESET} {m['content']}")
            else:
                print(f"{BLUE}AGENT>{RESET} {m['content']}")
        print("\n[AGENT streaming output]")

        return self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens, stream=True)


# -----------------------
# Quickstart demonstration
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_model", type=str, default=os.getenv("CHUTES_MODEL_AGENT", "deepseek-ai/DeepSeek-V3-0324"))
    parser.add_argument("--user_model", type=str, default=os.getenv("CHUTES_MODEL_USER", "deepseek-ai/DeepSeek-V3-0324"))
    parser.add_argument("--user_seed", type=str, default="Hello ! Can you describe what is bittensor ?")
    args = parser.parse_args()

    env_client = ChutesClient(api_key=os.getenv("CHUTES_API_TOKEN_USER", os.getenv("CHUTES_API_TOKEN")), model=args.user_model)
    env = ChutesConversationEnv(env_client, ChutesEnvConfig(max_turns=4))

    agent_client = ChutesClient(api_key=os.getenv("CHUTES_API_TOKEN_AGENT", os.getenv("CHUTES_API_TOKEN")), model=args.agent_model)
    policy = ChutesActorPolicy(client=agent_client)

    obs, info = env.reset(options={"user_seed": args.user_seed})
    done = False
    step_i = 0
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\n[step {step_i}] reward={reward:.3f}\n")
        done = terminated or truncated
        step_i += 1

    print("\nFinished.")
