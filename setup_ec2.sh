#!/bin/bash
# Quick setup script for EC2

echo "🚀 Setting up ML Product Pricing on EC2..."
echo ""

# Update system
echo "→ Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install Python 3.10 and dependencies
echo "→ Installing Python 3.10 and dependencies..."
sudo apt install -y python3 python3-pip python3-venv python3-dev
sudo apt install -y build-essential libssl-dev libffi-dev
sudo apt install -y git htop tmux

# Verify Python version
echo ""
echo "→ Python version:"
python3 --version

# Install AWS CLI (optional but useful)
echo ""
echo "→ Installing AWS CLI..."
sudo apt install -y awscli

# Create project directory
echo ""
echo "→ Creating project directory..."
mkdir -p ~/ml-projects
cd ~/ml-projects

echo ""
echo "✅ EC2 setup complete!"
echo ""
echo "Next steps:"
echo "1. Clone repository: git clone https://github.com/YOUR_USERNAME/amazon-ml-product-pricing.git"
echo "2. cd amazon-ml-product-pricing"
echo "3. Create venv: python3 -m venv venv"
echo "4. Activate: source venv/bin/activate"
echo "5. Install deps: pip install -r requirements.txt"
echo "6. Upload data files from local machine"
echo "7. Run training: python train.py --config config.yaml --optimize"
echo ""
echo "📚 See GITHUB_EC2_DEPLOYMENT.md for detailed instructions"
