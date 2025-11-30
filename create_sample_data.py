#!/usr/bin/env python3
"""
Create comprehensive sample bug data with MLOps feedback for testing BugFlow
"""
from app.database import SessionLocal
from app import models
import datetime
import random

def create_comprehensive_sample_data():
    db = SessionLocal()
    
    # Get users
    tester = db.query(models.User).filter(models.User.email == "tester1@example.com").first()
    pm = db.query(models.User).filter(models.User.email == "pm1@example.com").first()
    
    if not tester or not pm:
        print("❌ Users not found! Run init_users.py first.")
        return
    
    # Comprehensive bug data (30+ bugs)
    sample_bugs = [
        # Critical bugs
        {"title": "Production server down", "description": "Main production server crashed at 3 AM. All users affected. Database connection lost.", "project": "Infrastructure", "severity": "critical", "team": "DevOps", "status": "resolved"},
        {"title": "Payment gateway failure", "description": "Stripe payment processing completely down. No transactions going through since morning.", "project": "Payments", "severity": "critical", "team": "Backend", "status": "in_progress"},
        {"title": "Data breach security alert", "description": "Security scanner detected potential SQL injection vulnerability in user login endpoint.", "project": "Security", "severity": "critical", "team": "Backend", "status": "in_progress"},
        {"title": "Email system completely broken", "description": "All email notifications stopped working. SMTP server connection refused errors.", "project": "Notifications", "severity": "critical", "team": "Backend", "status": "open"},
        
        # High severity bugs
        {"title": "Login button crashes Chrome", "description": "Chrome 120 crashes when clicking login button on Windows 10. Happens 100% of the time.", "project": "WebApp", "severity": "high", "team": "Frontend", "status": "open"},
        {"title": "Mobile app crashes on iOS 17", "description": "App crashes when uploading profile pictures on iOS 17. Works on iOS 16.", "project": "MobileApp", "severity": "high", "team": "Mobile", "status": "in_progress"},
        {"title": "Database query timeout", "description": "Monthly sales reports taking 60+ seconds. Database queries need optimization.", "project": "Analytics", "severity": "high", "team": "Backend", "status": "open"},
        {"title": "Shopping cart loses items", "description": "Users report shopping cart randomly losing items after 5 minutes of inactivity.", "project": "Ecommerce", "severity": "high", "team": "Backend", "status": "open"},
        {"title": "File upload fails for large files", "description": "Files over 10MB fail to upload. Returns 413 error. Need to increase upload limit.", "project": "FileManager", "severity": "high", "team": "Backend", "status": "resolved"},
        {"title": "Search results completely wrong", "description": "Search functionality returning irrelevant results. Algorithm needs reindexing.", "project": "Search", "severity": "high", "team": "Backend", "status": "in_progress"},
        
        # Medium severity bugs
        {"title": "Slow page load times", "description": "Dashboard taking 8-10 seconds to load. Performance optimization needed.", "project": "WebApp", "severity": "medium", "team": "Frontend", "status": "open"},
        {"title": "Export CSV feature broken", "description": "CSV export generates empty files. Data is there but export functionality broken.", "project": "Reports", "severity": "medium", "team": "Backend", "status": "open"},
        {"title": "Dark mode inconsistencies", "description": "Some UI elements don't switch properly in dark mode. Text remains dark.", "project": "WebApp", "severity": "medium", "team": "Frontend", "status": "resolved"},
        {"title": "Notification badges not updating", "description": "Notification count badge not updating in real-time. Requires page refresh.", "project": "Notifications", "severity": "medium", "team": "Frontend", "status": "open"},
        {"title": "API rate limiting too aggressive", "description": "Users getting rate limited after only 50 requests. Limit should be 100.", "project": "API", "severity": "medium", "team": "Backend", "status": "in_progress"},
        {"title": "Profile picture upload slow", "description": "Profile picture uploads taking 30+ seconds. Image compression needed.", "project": "Profile", "severity": "medium", "team": "Backend", "status": "open"},
        {"title": "Broken links in footer", "description": "Privacy policy and Terms links in footer return 404 errors.", "project": "WebApp", "severity": "medium", "team": "Frontend", "status": "resolved"},
        {"title": "Chart data not refreshing", "description": "Analytics charts showing stale data from yesterday. Need to fix cache.", "project": "Analytics", "severity": "medium", "team": "Frontend", "status": "open"},
        {"title": "Mobile keyboard covers input", "description": "On mobile Safari, keyboard covers password input field during login.", "project": "MobileWeb", "severity": "medium", "team": "Frontend", "status": "open"},
        {"title": "Timezone display incorrect", "description": "All timestamps showing in UTC instead of user's local timezone.", "project": "WebApp", "severity": "medium", "team": "Backend", "status": "in_progress"},
        
        # Low severity bugs
        {"title": "Typo in settings page", "description": "Settings menu shows 'Prefrences' instead of 'Preferences'.", "project": "WebApp", "severity": "low", "team": "Frontend", "status": "open"},
        {"title": "Button alignment off by 2px", "description": "Submit button slightly misaligned on checkout page. Design inconsistency.", "project": "Ecommerce", "severity": "low", "team": "Frontend", "status": "resolved"},
        {"title": "Tooltip text truncated", "description": "Help tooltips cutting off text on narrow screens. Need responsive fix.", "project": "WebApp", "severity": "low", "team": "Frontend", "status": "open"},
        {"title": "Icon missing on Safari", "description": "User profile icon not displaying on Safari 16. Works on other browsers.", "project": "WebApp", "severity": "low", "team": "Frontend", "status": "open"},
        {"title": "Console warning messages", "description": "Browser console showing React warning about unique keys. Not affecting functionality.", "project": "WebApp", "severity": "low", "team": "Frontend", "status": "resolved"},
        {"title": "Placeholder text too light", "description": "Input placeholder text barely visible in light mode. Needs darker color.", "project": "WebApp", "severity": "low", "team": "Frontend", "status": "open"},
        {"title": "Loading spinner off-center", "description": "Loading animation slightly off-center on mobile devices.", "project": "MobileWeb", "severity": "low", "team": "Frontend", "status": "open"},
        {"title": "Inconsistent button sizes", "description": "Primary and secondary buttons have different heights. Should be uniform.", "project": "DesignSystem", "severity": "low", "team": "Frontend", "status": "resolved"},
        
        # Additional bugs to reach 30+
        {"title": "Docker container memory leak", "description": "Backend container memory usage growing continuously. Needs restart every 48 hours.", "project": "Infrastructure", "severity": "high", "team": "DevOps", "status": "in_progress"},
        {"title": "Redis cache not clearing", "description": "Cached data persisting after manual clear command. Redis configuration issue.", "project": "Cache", "severity": "medium", "team": "DevOps", "status": "open"},
        {"title": "API documentation outdated", "description": "Swagger docs showing old endpoints. Need to regenerate from latest code.", "project": "Documentation", "severity": "low", "team": "Backend", "status": "open"},
        {"title": "Android app orientation bug", "description": "App layout breaks when rotating device from portrait to landscape.", "project": "MobileApp", "severity": "medium", "team": "Mobile", "status": "open"},
        {"title": "Websocket connection drops", "description": "Real-time updates stop working after 10 minutes. Websocket timeout issue.", "project": "RealTime", "severity": "high", "team": "Backend", "status": "in_progress"},
    ]
    
    print(f"Creating {len(sample_bugs)} sample bugs...")
    
    created_bugs = []
    for i, bug_data in enumerate(sample_bugs):
        # Vary the creation time
        days_ago = random.randint(1, 30)
        hours_ago = random.randint(0, 23)
        
        bug = models.Bug(
            title=bug_data["title"],
            description=bug_data["description"],
            project=bug_data["project"],
            severity=bug_data["severity"],
            team=bug_data["team"],
            status=bug_data["status"],
            reporter_id=tester.id,
            created_at=datetime.datetime.utcnow() - datetime.timedelta(days=days_ago, hours=hours_ago),
            is_fake=False
        )
        db.add(bug)
        db.flush()  # Get the ID
        created_bugs.append(bug)
    
    db.commit()
    print(f"✅ Created {len(sample_bugs)} bugs")
    
    # Create MLOps feedback corrections for ~15 bugs (about 50%)
    print("\n📝 Creating MLOps feedback corrections...")
    
    feedback_scenarios = [
        # PM corrects severity that was predicted wrong
        {"bug_idx": 0, "correction_severity": "critical", "note": "Should be critical"},
        {"bug_idx": 4, "correction_severity": "critical", "note": "Login crash is critical"},
        {"bug_idx": 7, "correction_severity": "critical", "note": "Lost cart items = critical"},
        {"bug_idx": 10, "correction_team": "Backend", "note": "Backend performance issue"},
        {"bug_idx": 11, "correction_team": "Backend", "note": "CSV is backend logic"},
        
        # Team corrections
        {"bug_idx": 13, "correction_team": "Frontend", "note": "UI notification issue"},
        {"bug_idx": 17, "correction_team": "Backend", "note": "Cache is backend"},
        {"bug_idx": 18, "correction_team": "Mobile", "note": "Mobile-specific"},
        {"bug_idx": 28, "correction_severity": "critical", "correction_team": "DevOps", "note": "Memory leak critical"},
        
        # More corrections
        {"bug_idx": 5, "correction_severity": "critical", "note": "iOS crash critical"},
        {"bug_idx": 9, "correction_severity": "critical", "note": "Search is core feature"},
        {"bug_idx": 14, "correction_team": "Backend", "note": "API rate limit backend"},
        {"bug_idx": 29, "correction_team": "DevOps", "note": "Redis DevOps issue"},
        {"bug_idx": 31, "correction_severity": "high", "note": "Mobile layout important"},
        {"bug_idx": 32, "correction_severity": "critical", "correction_team": "Backend", "note": "Real-time critical"},
    ]
    
    feedback_count = 0
    for scenario in feedback_scenarios:
        if scenario['bug_idx'] < len(created_bugs):
            feedback = models.Feedback(
                bug_id=created_bugs[scenario['bug_idx']].id,
                correction_severity=scenario.get('correction_severity'),
                correction_team=scenario.get('correction_team'),
                created_at=datetime.datetime.utcnow() - datetime.timedelta(days=random.randint(0, 10))
            )
            db.add(feedback)
            feedback_count += 1
            
            # Update the bug with corrections
            if scenario.get('correction_severity'):
                created_bugs[scenario['bug_idx']].severity = scenario['correction_severity']
            if scenario.get('correction_team'):
                created_bugs[scenario['bug_idx']].team = scenario['correction_team']
    
    db.commit()
    
    print(f"✅ Created {feedback_count} MLOps feedback corrections")
    
    # Show statistics
    total_bugs = db.query(models.Bug).count()
    total_feedback = db.query(models.Feedback).count()
    
    print("\n" + "="*60)
    print("📊 DATABASE STATISTICS")
    print("="*60)
    print(f"Total Bugs: {total_bugs}")
    print(f"Total Feedback Corrections: {total_feedback}")
    print("\n🎯 Bug Status Breakdown:")
    print(f"  • Open: {db.query(models.Bug).filter(models.Bug.status == 'open').count()}")
    print(f"  • In Progress: {db.query(models.Bug).filter(models.Bug.status == 'in_progress').count()}")
    print(f"  • Resolved: {db.query(models.Bug).filter(models.Bug.status == 'resolved').count()}")
    
    print("\n⚡ Severity Breakdown:")
    print(f"  • Critical: {db.query(models.Bug).filter(models.Bug.severity == 'critical').count()}")
    print(f"  • High: {db.query(models.Bug).filter(models.Bug.severity == 'high').count()}")
    print(f"  • Medium: {db.query(models.Bug).filter(models.Bug.severity == 'medium').count()}")
    print(f"  • Low: {db.query(models.Bug).filter(models.Bug.severity == 'low').count()}")
    
    print("\n👥 Team Breakdown:")
    print(f"  • Backend: {db.query(models.Bug).filter(models.Bug.team == 'Backend').count()}")
    print(f"  • Frontend: {db.query(models.Bug).filter(models.Bug.team == 'Frontend').count()}")
    print(f"  • Mobile: {db.query(models.Bug).filter(models.Bug.team == 'Mobile').count()}")
    print(f"  • DevOps: {db.query(models.Bug).filter(models.Bug.team == 'DevOps').count()}")
    
    print("\n🔄 MLOps Status:")
    print(f"  • Bugs with corrections: {feedback_count}")
    print(f"  • Correction rate: {(feedback_count/total_bugs*100):.1f}%")
    print(f"  • Ready for retraining: {'YES ✅' if total_feedback >= 50 else f'NO ({50-total_feedback} more needed)'}")
    
    db.close()

if __name__ == "__main__":
    print("=" * 60)
    print("BugFlow - Creating Comprehensive Sample Data with MLOps")
    print("=" * 60)
    
    # First, clear existing sample data
    db = SessionLocal()
    print("🗑️  Clearing existing sample data...")
    db.query(models.Feedback).delete()
    db.query(models.Bug).delete()
    db.commit()
    db.close()
    
    create_comprehensive_sample_data()
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\n🚀 Next Steps:")
    print("1. Refresh your browser at http://localhost:8501")
    print("2. Login as Tester to see all bugs")
    print("3. Login as PM to view MLOps feedback corrections")
    print("4. Check the Kanban board and Analytics!")
