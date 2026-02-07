# GeniusPro Server

Production AI server powering the GeniusPro API platform.

## Infrastructure

| Component | Port | Description |
|-----------|------|-------------|
| **Ollama** | 11434 | LLM inference (`geniuspro-coder-v1`) |
| **API Gateway** | 8000 | Auth, proxy, usage logging |
| **Voice Server** | 8001 | Real-time speech-to-speech (internal) |
| **Nginx** | 80 | Reverse proxy for `api.geniuspro.io` |

**Hardware:** RTX 5090 (32GB VRAM), Ubuntu Server  
**Domain:** `api.geniuspro.io` via Cloudflare Tunnel

## API Endpoints

All endpoints at `https://api.geniuspro.io` require an API key via `X-API-Key` header or `Authorization: Bearer` header, except `/health`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (no auth) |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/v1/voice` | WebSocket | Real-time voice (S2S) |

## Auth Flow

1. Customer signs up on the platform (Supabase Auth)
2. Customer creates an API key from the dashboard
3. Gateway validates the key hash against Supabase `api_keys` table
4. Every request is logged to `usage_logs` for billing

## Server Management

```bash
# SSH in
ssh -F config geniuspro

# Service status
sudo systemctl status geniuspro-gateway
sudo systemctl status geniuspro-voice
sudo systemctl status ollama
sudo systemctl status nginx

# Logs
sudo journalctl -u geniuspro-gateway -f
sudo journalctl -u geniuspro-voice -f

# Restart
sudo systemctl restart geniuspro-gateway
sudo systemctl restart geniuspro-voice

# GPU
nvidia-smi
```

## Project Structure

```
geniuspro-server/
├── config                              # SSH config
├── README.md                           # This file
├── gateway/
│   ├── gateway.py                      # API gateway (FastAPI)
│   ├── nginx-geniuspro-api.conf        # Nginx site config
│   ├── geniuspro-gateway.service       # systemd unit
│   └── geniuspro-voice.service         # systemd unit
└── voice-server/
    ├── server.py                       # Voice S2S server (FastAPI)
    └── client.html                     # Web client for voice
```

## Remote Deployment

```bash
# Deploy gateway changes
scp -F config gateway/gateway.py geniuspro:~/geniuspro-gateway/
ssh -F config geniuspro "sudo systemctl restart geniuspro-gateway"

# Deploy voice server changes
scp -F config voice-server/server.py geniuspro:~/geniuspro-voice-server/
ssh -F config geniuspro "sudo systemctl restart geniuspro-voice"

# Deploy nginx changes
scp -F config gateway/nginx-geniuspro-api.conf geniuspro:~/geniuspro-gateway/
ssh -F config geniuspro "sudo cp ~/geniuspro-gateway/nginx-geniuspro-api.conf /etc/nginx/sites-available/geniuspro-api && sudo nginx -t && sudo systemctl reload nginx"
```

## Supabase (Platform DB)

Project: `orajwuisgwffnrbjasaj`

| Table | Purpose |
|-------|---------|
| `user_profiles` | Accounts, plans, credits |
| `api_keys` | Hashed API keys with rate limits |
| `usage_logs` | Request tracking for billing |
| `chat_sessions` | Chat UI conversations |
| `chat_messages` | Chat messages |
