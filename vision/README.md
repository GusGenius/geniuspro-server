# GeniusPro Vision Service

SAM 3 (Segment Anything Model 3) integration for image and video segmentation with text prompts.

## Overview

The Vision Service provides REST API endpoints for:
- **Image Segmentation**: Segment objects in static images using **text prompts** (e.g., "a red car", "player in white"), point prompts, or box prompts
- **Video Segmentation**: Track and segment objects across video frames with temporal consistency using text or visual prompts

## Key Features

- **Text Prompts**: SAM 3's main feature - segment objects using natural language descriptions
- **Open-Vocabulary**: Handles 270K+ unique concepts, not limited to training data
- **Better Performance**: 37.2 cgF1 on SA-Co/Gold vs 29.3 for OWLv2
- **Visual Prompts**: Still supports points and boxes like SAM 2

## Installation

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA 12.6 or higher (for GPU support)
- Hugging Face account with access to SAM 3 checkpoints

### 1. Request SAM 3 Checkpoint Access

SAM 3 checkpoints require Hugging Face authentication:

1. Visit https://huggingface.co/facebook/sam3-hiera-large
2. Request access to the SAM 3 model repository
3. Generate a Hugging Face access token: https://huggingface.co/settings/tokens
4. Set environment variable: `export HUGGING_FACE_TOKEN=your_token_here`

### 2. Install SAM 3

```bash
# Clone SAM 3 repository
cd ~
git clone https://github.com/facebookresearch/sam3.git
cd sam3

# Install SAM 3
pip install -e .
```

### 3. Setup Vision Service

```bash
# Create directory structure
mkdir -p ~/geniuspro-vision/vision
cd ~/geniuspro-vision

# Create virtual environment (Python 3.12+ required)
python3.12 -m venv venv
source venv/bin/activate

# Install vision service dependencies
pip install -r vision/requirements.txt
```

### 4. Configuration

Set environment variables in `~/geniuspro-vision/.env`:

```bash
# Supabase (for auth)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key

# SAM 3 Configuration
SAM3_DEVICE=cuda  # "cuda" or "cpu"
HUGGING_FACE_TOKEN=your_hf_token_here  # Required for checkpoint access
```

### 5. Run the Service

**Development:**
```bash
cd ~/geniuspro-vision
source venv/bin/activate
uvicorn vision.app:app --host 0.0.0.0 --port 8200
```

**Production (systemd):**
```bash
# Copy service file
sudo cp ~/geniuspro-vision/geniuspro-vision.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable geniuspro-vision
sudo systemctl start geniuspro-vision

# Check status
sudo systemctl status geniuspro-vision
```

### 6. Nginx Configuration

Add the vision service location block to your nginx config:

```nginx
location /vision/ {
    proxy_pass http://127.0.0.1:8200/vision/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    
    # Increase timeouts for video processing
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    
    # Increase body size for video uploads
    client_max_body_size 500M;
}
```

## API Endpoints

### Health Check

```bash
GET /vision/v1/health
```

### List Models

```bash
GET /vision/v1/models
Authorization: Bearer <api-key>
```

### Image Segmentation

**With Text Prompt (SAM 3's main feature):**
```bash
POST /vision/v1/segment-image
Authorization: Bearer <api-key>
Content-Type: multipart/form-data

Form fields:
- image: (file) Image file (PNG, JPEG, etc.)
- text_prompt: (string) Text description (e.g., "a red car", "player in white")
- input_points: (optional, JSON string) [[x1, y1], [x2, y2]]
- input_labels: (optional, JSON string) [1, 0]
- input_box: (optional, JSON string) [x1, y1, x2, y2]

Example with curl:
curl -X POST http://localhost:8200/vision/v1/segment-image \
  -H "Authorization: Bearer sk-gp-..." \
  -F "image=@image.jpg" \
  -F "text_prompt=a red car"
```

**With Visual Prompts (points/box):**
```bash
curl -X POST http://localhost:8200/vision/v1/segment-image \
  -H "Authorization: Bearer sk-gp-..." \
  -F "image=@image.jpg" \
  -F "input_points=[[100, 200]]" \
  -F "input_labels=[1]"
```

### Video Segmentation

**1. Initialize video session:**
```bash
POST /vision/v1/segment-video/init
Authorization: Bearer <api-key>
Content-Type: multipart/form-data

Form fields:
- video: (file) Video file (MP4) or directory of JPEG frames

Returns: { "session_id": "uuid-string", "video_path": "..." }
```

**2. Add prompts (text or visual):**
```bash
POST /vision/v1/segment-video/add-prompt
Authorization: Bearer <api-key>
Content-Type: multipart/form-data

Form fields:
- session_id: (string) Session ID from init
- text_prompt: (optional, string) Text description (e.g., "a red car")
- frame_index: (optional, int) Frame index (default: 0)
- input_points: (optional, JSON string) [[x1, y1], [x2, y2]]
- input_labels: (optional, JSON string) [1, 0]
- input_box: (optional, JSON string) [x1, y1, x2, y2]
```

**3. Propagate masks:**
```bash
POST /vision/v1/segment-video/propagate
Authorization: Bearer <api-key>
Content-Type: multipart/form-data

Form fields:
- session_id: (string) Session ID from init

Returns: { "frames": [{ "frame_index": 0, "masks": [...], "boxes": [...], "scores": [...] }, ...] }
```

**4. Cleanup:**
```bash
POST /vision/v1/segment-video/cleanup
Authorization: Bearer <api-key>
Content-Type: multipart/form-data

Form fields:
- session_id: (string) Session ID to cleanup
```

## Model Performance

SAM 3 achieves state-of-the-art performance:

| Task | Metric | SAM 3 | Previous Best |
|------|--------|-------|---------------|
| SA-Co/Gold | cgF1 | 37.2 | 29.3 (OWLv2) |
| SA-Co/Gold | AP | 48.5 | 43.4 (OWLv2) |
| LVIS | cgF1 | 54.1 | 24.6 (OWLv2) |
| COCO | AP | 53.6 | 45.5 (OWLv2) |

## Troubleshooting

### Hugging Face Authentication Error
- Ensure you've requested access to the SAM 3 checkpoint repository
- Set `HUGGING_FACE_TOKEN` environment variable
- Login via CLI: `huggingface_hub login` or `hf auth login`

### Python Version Error
- SAM 3 requires Python 3.12+
- Check version: `python3 --version`
- Create new venv with Python 3.12: `python3.12 -m venv venv`

### PyTorch/CUDA Version Error
- SAM 3 requires PyTorch 2.7+ and CUDA 12.6+
- Install: `pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
- Verify CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`

### Import Errors
- Ensure SAM 3 is installed: `pip install -e "git+https://github.com/facebookresearch/sam3.git"`
- Verify installation: `python3 -c "import sam3; print(sam3.__version__)"`

## Architecture

```
vision/
├── app.py              # FastAPI application
├── config.py           # Configuration loader
├── sam3_predictor.py   # SAM 3 wrapper
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── geniuspro-vision.service  # systemd service
└── nginx-vision.conf   # Nginx config snippet
```

## License

SAM 3 is licensed under the SAM License. See the [SAM 3 repository](https://github.com/facebookresearch/sam3) for details.
