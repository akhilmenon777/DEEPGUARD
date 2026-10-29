# Vercel Deployment Guide for Deepfake Detection System

This guide will help you deploy your deepfake detection system to Vercel with serverless Python functions.

## 📋 Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI** (optional): `npm install -g vercel`
3. **Git Repository**: Your code should be in a Git repository (GitHub, GitLab, or Bitbucket)

## 🚀 Quick Deploy (Recommended Method)

### Option 1: Deploy via Vercel Dashboard

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/deepfake-detection.git
   git push -u origin main
   ```

2. **Import to Vercel**:
   - Go to [vercel.com/new](https://vercel.com/new)
   - Click "Import Project"
   - Select your GitHub repository
   - Vercel will auto-detect the configuration

3. **Configure Environment** (if needed):
   - Click "Environment Variables" (optional)
   - Add any required environment variables
   - Click "Deploy"

4. **Wait for deployment**:
   - First deployment takes ~5-10 minutes
   - Subsequent deployments are faster

### Option 2: Deploy via Vercel CLI

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   cd deepfake-detection-system
   vercel
   ```

4. **Follow the prompts**:
   - Set up and deploy: `Y`
   - Which scope: Select your account
   - Link to existing project: `N`
   - Project name: `deepfake-detection`
   - Directory: `./` (default)
   - Override settings: `N`

5. **Production deployment**:
   ```bash
   vercel --prod
   ```

## 📁 Project Structure for Vercel

The project is structured for Vercel deployment:

```
deepfake-detection-system/
├── api/                        # Serverless Python functions
│   ├── __init__.py
│   ├── config.py
│   ├── index.py               # Health check endpoint
│   ├── predict-image.py       # Image detection endpoint
│   ├── predict-video.py       # Video detection endpoint
│   ├── predict-audio.py       # Audio detection endpoint
│   └── requirements.txt       # Python dependencies
├── backend/                    # Core detection logic
│   ├── detectors/
│   ├── models/
│   └── utils/
├── frontend/                   # Static frontend files
│   ├── index.html
│   ├── css/
│   └── js/
├── weights/                    # Model weights
│   ├── gend_vit_best.pth
│   └── audio_cnn_best.pth
└── vercel.json                # Vercel configuration
```

## ⚙️ Configuration Files

### vercel.json

The `vercel.json` file configures:
- Python serverless functions in `/api`
- Static file serving for frontend
- Function memory (3GB) and timeout (60s)
- Route mapping

### API Routes

After deployment, your API will be available at:
- `https://your-project.vercel.app/api` - Health check
- `https://your-project.vercel.app/api/predict-image` - Image detection
- `https://your-project.vercel.app/api/predict-video` - Video detection
- `https://your-project.vercel.app/api/predict-audio` - Audio detection

## 🔧 Important Notes

### Model Weights

**⚠️ IMPORTANT**: Vercel has deployment size limits:
- Free tier: 100MB total
- Pro tier: 250MB per function

Your model weights might exceed these limits. Solutions:

#### Solution 1: Use External Storage (Recommended)

Store weights externally and load them at runtime:

1. Upload weights to cloud storage (AWS S3, Google Cloud Storage, etc.)
2. Update `api/config.py`:

```python
import os
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path("/tmp/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

def download_weights():
    """Download weights from external URL if not present"""
    if not (WEIGHTS_DIR / "gend_vit_best.pth").exists():
        urllib.request.urlretrieve(
            "YOUR_S3_URL/gend_vit_best.pth",
            str(WEIGHTS_DIR / "gend_vit_best.pth")
        )
    if not (WEIGHTS_DIR / "audio_cnn_best.pth").exists():
        urllib.request.urlretrieve(
            "YOUR_S3_URL/audio_cnn_best.pth",
            str(WEIGHTS_DIR / "audio_cnn_best.pth")
        )

GEND_WEIGHTS_PATH = WEIGHTS_DIR / "gend_vit_best.pth"
AUDIO_CNN_WEIGHTS_PATH = WEIGHTS_DIR / "audio_cnn_best.pth"
```

#### Solution 2: Use Smaller Models

Train/use quantized or smaller versions of your models.

#### Solution 3: Separate Deployment

Deploy each detector as a separate Vercel project:
- `deepfake-image-api` - Image/Video detection
- `deepfake-audio-api` - Audio detection

### Cold Starts

Serverless functions have "cold starts" (1-5 seconds) when not used recently. The first request after idle time will be slower.

**Optimization tips**:
- Use lazy loading (already implemented)
- Consider Vercel Pro for faster cold starts
- Keep functions warm with scheduled pings (cron job)

### Function Limits

Vercel serverless function limits:
- **Execution time**: 60 seconds (configured in vercel.json)
- **Memory**: 3GB (configured in vercel.json)
- **Payload size**: 4.5MB request body

For large videos/audio files, consider:
- Client-side compression
- Chunked upload
- External processing queue

## 🧪 Testing Your Deployment

1. **Test health endpoint**:
   ```bash
   curl https://your-project.vercel.app/api/health
   ```

2. **Test image prediction**:
   ```bash
   curl -X POST https://your-project.vercel.app/api/predict-image \
     -F "file=@sample_image.jpg"
   ```

3. **Open the web interface**:
   ```
   https://your-project.vercel.app
   ```

## 🔄 Continuous Deployment

Once connected to GitHub:
- Every `git push` to main triggers automatic deployment
- Pull requests get preview deployments
- Rollback to previous deployments in Vercel dashboard

## 🐛 Troubleshooting

### Build Fails

1. Check build logs in Vercel dashboard
2. Verify `requirements.txt` is correct
3. Ensure all imports work

### Function Timeout

- Reduce MAX_FRAMES for video processing
- Optimize model inference
- Increase timeout in vercel.json (max 300s on Pro)

### Memory Issues

- Use smaller batch sizes
- Clear tensors after use: `torch.cuda.empty_cache()`
- Increase memory limit in vercel.json

### 404 Errors

- Verify route paths in `vercel.json`
- Check function names match file names
- Ensure `handler` function is exported

## 📊 Monitoring

Monitor your deployment:
1. Go to Vercel Dashboard
2. Select your project
3. View:
   - Function logs
   - Performance metrics
   - Error tracking
   - Analytics

## 💰 Cost Considerations

**Free Tier**:
- 100GB bandwidth
- Serverless function executions
- Usually sufficient for testing/demos

**Pro Tier** ($20/month):
- Faster builds and cold starts
- More function execution time
- Priority support

## 🔐 Security

1. **API Keys**: Store in Vercel Environment Variables
2. **CORS**: Configure in your API files
3. **Rate Limiting**: Consider adding rate limiting middleware
4. **Input Validation**: Already implemented in the code

## 📚 Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Python on Vercel](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- [Vercel CLI Reference](https://vercel.com/docs/cli)

## ✅ Success Checklist

- [ ] Code pushed to Git repository
- [ ] Model weights hosted or included
- [ ] Deployed to Vercel
- [ ] Health endpoint responds
- [ ] Image prediction works
- [ ] Video prediction works (if weights included)
- [ ] Audio prediction works (if weights included)
- [ ] Frontend loads correctly
- [ ] CORS configured properly
- [ ] Monitoring set up

## 🆘 Support

If you encounter issues:
1. Check Vercel function logs
2. Review this guide
3. Check Vercel community forums
4. Contact Vercel support (Pro tier)

---

**Ready to deploy?** Run `vercel --prod` and share your deepfake detection system with the world! 🚀
