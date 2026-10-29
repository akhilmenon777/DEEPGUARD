# 🚀 Quick Start Guide - Deploy in 5 Minutes

This is the fastest way to get your deepfake detection system live on Vercel.

## Prerequisites (1 minute)

- [ ] GitHub account ([signup](https://github.com/signup))
- [ ] Vercel account ([signup](https://vercel.com/signup))
- [ ] Git installed on your computer

## Step 1: Push to GitHub (2 minutes)

Open terminal in the project folder:

```bash
# Initialize Git repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Deepfake Detection System"

# Create main branch
git branch -M main

# Add your GitHub repository
# (Create a new empty repo on GitHub first, then copy its URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push
git push -u origin main
```

## Step 2: Deploy to Vercel (2 minutes)

### Via Dashboard (Easiest):

1. Go to [vercel.com/new](https://vercel.com/new)
2. Click **"Import Git Repository"**
3. Select your GitHub repository
4. Click **"Import"**
5. Vercel auto-detects the configuration
6. Click **"Deploy"**
7. Wait ~5 minutes ☕
8. Done! Your app is live!

### Via CLI (Alternative):

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
cd deepfake-detection-system
vercel --prod
```

## Step 3: Test Your Deployment (30 seconds)

Visit your deployment URL (shown after deployment):
```
https://your-project.vercel.app
```

Try the preset buttons:
- 📷 **Authentic Face** - Should detect as real
- ⚠️ **Deepfake Face** - Should detect as fake
- 🎙️ **Genuine Speech** - Should detect as real
- ⚡ **Voice Clone** - Should detect as synthetic

## ⚠️ Important Notes

### Model Weights

Your `.pth` model files are large. If deployment fails due to size limits:

**Quick Fix**: Deploy without weights (for testing UI/API):
```bash
# Temporarily exclude weights
echo "weights/*.pth" >> .gitignore
git add .gitignore
git commit -m "Exclude weights for testing"
git push
```

**Production Fix**: Upload weights to cloud storage:
1. Upload `weights/*.pth` to AWS S3, Google Cloud Storage, or Dropbox
2. Get public URLs
3. Add environment variables in Vercel Dashboard:
   - `GEND_WEIGHTS_URL` = your URL
   - `AUDIO_WEIGHTS_URL` = your URL
4. See `VERCEL_DEPLOYMENT.md` for code changes

## 🎯 What You Get

After deployment, you'll have:
- ✅ Live web interface at `https://your-project.vercel.app`
- ✅ API endpoints at `https://your-project.vercel.app/api/`
- ✅ Automatic deployments on every git push
- ✅ Preview deployments for pull requests
- ✅ Free SSL certificate
- ✅ Global CDN distribution

## 📱 Share Your App

Your deployment URL: `https://your-project-name.vercel.app`

Custom domain: Configure in Vercel Dashboard → Settings → Domains

## 🐛 Troubleshooting

### "Build Failed"
- Check build logs in Vercel Dashboard
- Ensure `api/requirements.txt` is correct
- Try deploying without weights first

### "Function Timeout"
- First deployment takes longer (5-10 min)
- Check function logs in Vercel Dashboard
- See `VERCEL_DEPLOYMENT.md` for optimization tips

### "Cannot find module"
- Verify all imports in your Python files
- Check that `backend/` folder is included
- Ensure `sys.path.insert(0, ...)` is correct in API files

## 📚 Next Steps

1. ✅ Read `README_VERCEL.md` for full API documentation
2. ✅ Review `VERCEL_DEPLOYMENT.md` for advanced configuration
3. ✅ Set up custom domain (optional)
4. ✅ Configure environment variables (if needed)
5. ✅ Share your deployment!

## 💡 Pro Tips

- **Continuous Deployment**: Every push to `main` auto-deploys
- **Preview URLs**: PRs get temporary preview URLs
- **Rollback**: Easy rollback in Vercel Dashboard
- **Monitoring**: Check logs and analytics in Dashboard

## 🎉 That's It!

You now have a live, scalable deepfake detection system!

**Your deployment URL**: `https://your-project.vercel.app`

---

Need help? Check `VERCEL_DEPLOYMENT.md` or visit [Vercel Docs](https://vercel.com/docs)
