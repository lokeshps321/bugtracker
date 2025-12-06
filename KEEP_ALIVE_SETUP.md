# Keep-Alive Setup for Render.com Free Tier

Render's free tier puts your backend to sleep after 15 minutes of inactivity.
This causes 30-60 second cold starts when users try to login.

## Solution: External Cron Ping

Use a **free cron service** to ping your backend every 5-10 minutes.

### Option 1: cron-job.org (Recommended)

1. Go to [cron-job.org](https://cron-job.org) and create a free account
2. Click **"Create cronjob"**
3. Fill in:
   - **Title**: BugFlow Keep Alive
   - **URL**: `https://bugflow.onrender.com/health`
   - **Schedule**: Every 5 minutes (select "Every 5 minutes" from dropdown)
4. Click **Save**

That's it! Your backend will stay awake 24/7.

### Option 2: UptimeRobot (Alternative)

1. Go to [uptimerobot.com](https://uptimerobot.com) and sign up
2. Click **"Add New Monitor"**
3. Select **HTTP(s)**
4. Fill in:
   - **Friendly Name**: BugFlow Backend  
   - **URL**: `https://bugflow.onrender.com/health`
   - **Monitoring Interval**: 5 minutes
5. Click **Create Monitor**

### Built-in Fallback

The Streamlit frontend now includes automatic keep-alive pings:
- When any user visits the app, it pings `/health` in the background
- This helps warm up the backend before login attempts

## Verify It Works

After setting up, you can check:
1. Visit `https://bugflow.onrender.com/health`
2. Should return: `{"status":"healthy","service":"bugflow-backend"}`

If the response is fast (< 1 second), the backend is awake!
