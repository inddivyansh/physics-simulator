# 🚀 GitHub Deployment Guide for Streamlit Community Cloud

## Step 1: Initialize Git Repository

Open a terminal/command prompt in your project folder and run:

```bash
cd "d:\Standard\Projects\Physics_Simulator"
git init
git add .
git commit -m "Initial commit: Physics Simulator with Streamlit app"
```

## Step 2: Create GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" button in the top right
3. Select "New repository"
4. Name it: `physics-simulator` (or your preferred name)
5. Make it **Public** (required for free Streamlit deployment)
6. Don't add README, .gitignore, or license (we already have them)
7. Click "Create repository"

## Step 3: Connect Local Repository to GitHub

Copy the commands from GitHub's "push an existing repository" section:

```bash
git remote add origin https://github.com/YOUR_USERNAME/physics-simulator.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 4: Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/physics-simulator`
5. Set the main file path: `streamlit_app.py`
6. Click "Deploy!"

## Step 5: Wait for Deployment

- The app will take a few minutes to deploy
- You'll see build logs showing the installation process
- Once complete, you'll get a public URL like: `https://your-app-name.streamlit.app`

## Troubleshooting

### If deployment fails:

1. **Check the build logs** for specific error messages
2. **Common issues:**
   - Missing dependencies in `requirements.txt`
   - Import path problems
   - File not found errors

### If you see import errors:

The app structure is designed to work with Streamlit Cloud's deployment system. The imports should resolve correctly in the cloud environment.

### To update your app:

```bash
git add .
git commit -m "Update: description of changes"
git push
```

The app will automatically redeploy when you push changes to GitHub.

## File Structure for Deployment

```
physics-simulator/
├── streamlit_app.py              # ⭐ Main entry point for Streamlit
├── requirements.txt              # ⭐ Required dependencies  
├── README.md                     # Project description
├── .gitignore                    # Git ignore rules
├── project/                      # Source code
│   ├── emergent_simulator.py     # Core simulation
│   ├── streamlit_app/
│   ├── notebooks/
│   └── ...
```

## Key Files for Deployment:

- **`streamlit_app.py`** - Main entry point (must be at root level)
- **`requirements.txt`** - All Python dependencies
- **`README.md`** - Project documentation for GitHub

## Success! 🎉

Once deployed, your physics simulator will be available at:
`https://your-app-name.streamlit.app`

Share the link with others to showcase your interactive physics simulation!

## Next Steps

1. Add the deployment URL to your README
2. Share your project on social media
3. Consider adding more features or physics rules
4. Explore Streamlit's advanced features

---

*Need help? Check the [Streamlit Community Cloud docs](https://docs.streamlit.io/streamlit-community-cloud) or ask in the [Streamlit community forum](https://discuss.streamlit.io/).*
