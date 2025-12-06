"""
Report 15 bugs directly to PostgreSQL cloud database
"""
import os
import psycopg2
from datetime import datetime

# Cloud database URL from render
DATABASE_URL = "postgresql://bugflow_db_user:hA31vvKBRwm0LSlSYz0Z5cBFaYLjrDC3@dpg-ctaqjfbtq21c738l9j50-a.oregon-postgres.render.com/bugflow_db"

# 15 diverse bug reports covering all teams and severities
BUGS = [
    # DevOps bugs (3)
    ("CI/CD pipeline failing on production deployment", "The Jenkins pipeline is failing during the Docker build stage. Error: Cannot connect to Docker daemon. This is blocking all deployments to production.", "WebApp", "high", "DevOps"),
    ("Kubernetes pods in CrashLoopBackOff state", "Multiple pods are crashing on startup with OOMKilled errors. The cluster autoscaler is not responding to increased load.", "WebApp", "critical", "DevOps"),
    ("Staging database sync failing nightly", "The automated script that copies production data to staging is aborting midway. Blocking QA team from testing.", "WebApp", "high", "DevOps"),
    
    # Mobile bugs (3)
    ("iOS app crashes on iPhone 14 when opening camera", "Users report the app force closes when using camera for QR scanning. Crash logs show nil pointer exception.", "MobileApp", "high", "Mobile"),
    ("Android push notifications not received on Samsung", "Push notifications not delivered to Samsung Galaxy devices running Android 13. FCM token registration succeeds but no messages.", "MobileApp", "high", "Mobile"),
    ("React Native app battery drain issue", "Users report 30% battery usage per hour in background. Location services and background fetch need optimization.", "MobileApp", "medium", "Mobile"),
    
    # Backend bugs (3)
    ("REST API returning 500 errors on high load", "The /api/orders endpoint throws errors when concurrent requests exceed 100. Database connection pool exhausted.", "WebApp", "high", "Backend"),
    ("JWT token validation failing after server restart", "Users logged out unexpectedly after deployments. JWT secret key rotation not preserving valid tokens.", "WebApp", "high", "Backend"),
    ("GraphQL resolver N+1 query issue", "The userOrders query makes 1000+ database calls per request. DataLoader implementation needed.", "WebApp", "medium", "Backend"),
    
    # Frontend bugs (3)
    ("Modal dialog not closing when clicking outside", "Checkout confirmation modal stays open on outside click. Users must click X button to close.", "WebApp", "low", "Frontend"),
    ("Dark mode toggle not persisting after refresh", "Dark mode preference resets to light mode after browser refresh. LocalStorage not saving.", "WebApp", "low", "Frontend"),
    ("React component not re-rendering on state change", "Shopping cart count badge doesn't update when adding items. Requires page refresh.", "WebApp", "medium", "Frontend"),
    
    # Mixed bugs (3)
    ("Typo in checkout success message", "Confirmation shows 'Congradulations' instead of 'Congratulations'. Minor cosmetic issue.", "WebApp", "low", "Frontend"),
    ("User session expiring too frequently", "Users logged out after 5 minutes. Session timeout should be 30 minutes per requirements.", "WebApp", "high", "Backend"),
    ("Search results page loading slowly", "Product search takes 8-10 seconds. Users abandoning searches. Need Elasticsearch optimization.", "WebApp", "medium", "Backend"),
]

def report_bugs():
    print(f"🐛 Reporting {len(BUGS)} bugs to cloud PostgreSQL...")
    print("-" * 50)
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        success = 0
        for i, (title, desc, project, severity, team) in enumerate(BUGS, 1):
            try:
                cur.execute("""
                    INSERT INTO bugs (title, description, project, severity, assigned_team, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, 'open', %s)
                """, (title, desc, project, severity, team, datetime.now()))
                
                print(f"✅ [{i}/15] {title[:45]}... -> {severity}/{team}")
                success += 1
            except Exception as e:
                print(f"❌ [{i}/15] Error: {str(e)[:50]}")
        
        conn.commit()
        cur.close()
        conn.close()
        
        print("-" * 50)
        print(f"✅ Successfully reported {success}/{len(BUGS)} bugs!")
        
    except Exception as e:
        print(f"❌ Database connection error: {e}")

if __name__ == "__main__":
    report_bugs()
