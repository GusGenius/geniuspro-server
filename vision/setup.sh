#!/bin/bash
# Setup script for GeniusPro Vision Service

set -e

echo "GeniusPro Vision Service Setup"
echo "=============================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "ERROR: Python 3.10+ required. Found: $python_version"
    exit 1
fi

echo "✓ Python version: $python_version"

# Check if SAM 2 is installed
if ! python3 -c "import sam2" 2>/dev/null; then
    echo ""
    echo "SAM 2 not found. Installing..."
    echo ""
    
    # Check if sam2 directory exists in parent
    if [ ! -d "../sam2" ]; then
        echo "Cloning SAM 2 repository..."
        cd ..
        git clone https://github.com/facebookresearch/sam2.git
        cd sam2
    else
        echo "SAM 2 directory found, using existing..."
        cd ../sam2
    fi
    
    echo "Installing SAM 2..."
    pip install -e .
    cd ../geniuspro-server/vision
    echo "✓ SAM 2 installed"
else
    echo "✓ SAM 2 already installed"
fi

# Install vision service dependencies
echo ""
echo "Installing vision service dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create checkpoints directory
echo ""
echo "Setting up checkpoints directory..."
echo "Note: On the server, checkpoints should be in ~/geniuspro-vision/checkpoints/"
mkdir -p checkpoints
cd checkpoints

# Check if checkpoints exist
if [ ! -f "sam2.1_hiera_large.pt" ]; then
    echo ""
    echo "Downloading SAM 2.1 checkpoints..."
    echo "This may take a while..."
    
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt || echo "Failed to download tiny"
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt || echo "Failed to download small"
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt || echo "Failed to download base_plus"
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt || echo "Failed to download large"
    
    echo "✓ Checkpoints downloaded"
else
    echo "✓ Checkpoints already exist"
fi

cd ../vision

echo ""
echo "=============================="
echo "Local setup complete!"
echo ""
echo "For server deployment:"
echo "1. Deploy files to ~/geniuspro-vision/vision/ on server"
echo "2. Create venv: python3 -m venv ~/geniuspro-vision/venv"
echo "3. Install SAM 2 and dependencies (see README.md)"
echo "4. Download checkpoints to ~/geniuspro-vision/checkpoints/"
echo "5. Create .env file with SUPABASE_URL, SUPABASE_SERVICE_KEY, etc."
echo "6. Copy geniuspro-vision.service to /etc/systemd/system/"
echo "7. Enable and start: sudo systemctl enable geniuspro-vision && sudo systemctl start geniuspro-vision"
echo ""
echo "See README.md for complete server setup instructions."
