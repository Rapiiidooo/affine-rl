Chutes RL – Multi‑Turn Environment Template

This template spins up a *multi‑turn conversational RL environment* that uses a **Chutes** (SN64) model as the *environment/user side*, and a **second Chutes model** as the *assistant/agent side* (self‑play). Swap the assistant policy with your RL policy later if desired.

- **Env side**: forwards the transcript to a Chutes‑hosted model to get the *user/opponent* reply and yields observations + rewards.
- **Agent side**: a Chutes‑powered policy (`ChutesActorPolicy`) that returns the assistant message (action).

⚠️ Notes
- Endpoints and auth headers may change. Adjust `ChutesClient.BASE_URL` and headers per official Chutes docs.
- Uses `gymnasium` and `spaces.Text`. Use `TextToIDsWrapper` if your RL lib requires numeric spaces.

Dependencies

`.env`
```bash
CHUTES_API_TOKEN=  # (Required)
# CHUTES_API_TOKEN_USER=  # (Optional)
# CHUTES_API_TOKEN_AGENT=  # (Optional)
# CHUTES_MODEL_AGENT=  # (Optional)
# CHUTES_MODEL_USER=  # (Optional)
```

```bash
pip install -r requirements.txt 
```
(Optional) If you plan to train with token IDs:
```bash
pip install transformers
```

# Quickstart
```bash
python affine_rl.py --user_model deepseek-ai/DeepSeek-V3-0324 --agent_model deepseek-ai/DeepSeek-V3-0324
python affine_rl.py --user_model deepseek-ai/DeepSeek-V3-0324 --agent_model deepseek-ai/DeepSeek-V3-0324 --user_seed "Tell me something about bittensor ?"
```