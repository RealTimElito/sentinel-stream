# Troubleshooting Guide

## Common Installation Issues

### pcap.h not found / pypcap installation fails

**Problem**: `pypcap` requires libpcap development headers to compile.

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install libpcap-dev

# macOS
brew install libpcap

# Or use the automated script
bash scripts/install_system_deps.sh
```

**Note**: If you can't install libpcap-dev, the system will fall back to using `scapy` for packet capture, which doesn't require system libraries.

### Conda environment creation fails

**Problem**: Some packages fail to install during `conda env create`.

**Solutions**:
1. Update conda: `conda update -n base -c defaults conda`
2. Try installing packages individually:
   ```bash
   conda create -n sentinel-stream python=3.9
   conda activate sentinel-stream
   conda install pytorch numpy pandas scikit-learn -c pytorch -c conda-forge
   pip install -r requirements.txt
   ```

### Redis connection errors

**Problem**: Cannot connect to Redis.

**Solutions**:
1. Check if Redis is running: `redis-cli ping` (should return `PONG`)
2. Start Redis:
   ```bash
   # Ubuntu/Debian
   sudo systemctl start redis-server
   
   # macOS
   brew services start redis
   
   # Docker
   docker run -d -p 6379:6379 redis:7-alpine
   ```
3. Check Redis host/port in config files (`configs/inference_config.yaml`)

### Packet capture requires root/sudo

**Problem**: `Permission denied` when running packet capture.

**Solution**: Packet capture requires elevated privileges:
```bash
sudo python scripts/capture_live.py --interface eth0
```

Or run as root user (not recommended for production).

### Model file not found

**Problem**: `FileNotFoundError` when starting inference API.

**Solution**: Train the model first:
```bash
python scripts/train_model.py --config configs/tgn_config.yaml
```

Or download a pre-trained model and place it in `models/tgn_model.pt`.

### Port already in use

**Problem**: Port 8000 or 3000 already in use.

**Solutions**:
1. Find and kill the process:
   ```bash
   # Find process using port 8000
   lsof -i :8000
   # Kill it
   kill -9 <PID>
   ```
2. Change ports in config files:
   - `configs/inference_config.yaml` for API port
   - `dashboard/package.json` for dashboard port

### CUDA/GPU issues

**Problem**: PyTorch can't find CUDA or GPU.

**Solutions**:
1. Install CUDA-enabled PyTorch:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
2. Check GPU availability:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
3. Set device to CPU in config if GPU not available:
   ```yaml
   # configs/inference_config.yaml
   model:
     device: "cpu"
   ```

### Import errors

**Problem**: `ModuleNotFoundError` or `ImportError`.

**Solutions**:
1. Ensure environment is activated: `conda activate sentinel-stream`
2. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Check Python path:
   ```bash
   python -c "import sys; print(sys.path)"
   ```

### Docker build fails

**Problem**: Docker build errors.

**Solutions**:
1. Check Docker is running: `docker ps`
2. Increase Docker memory limit (Docker Desktop → Settings → Resources)
3. Build without cache:
   ```bash
   docker-compose build --no-cache
   ```

## Getting Help

If you encounter issues not covered here:

1. Check the logs:
   - Application logs: `logs/`
   - Docker logs: `docker-compose logs`
   - System logs: `journalctl -u redis` (if using systemd)

2. Verify system requirements:
   - Python 3.9+
   - Redis installed and running
   - libpcap-dev installed (for packet capture)
   - Sufficient disk space and memory

3. Create an issue on GitHub with:
   - Error messages
   - System information (`uname -a`)
   - Python version (`python --version`)
   - Steps to reproduce

