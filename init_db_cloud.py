"""
Run this script ONCE to create database tables and initialize with demo users.
For cloud deployment without shell access.
"""
from app.database import Base, engine, SessionLocal
from app.models import User, Bug, MLOpsFeedback
from passlib.context import CryptContext

def init_database():
    """Create all tables and add demo users"""
    print("Creating database tables...")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created!")
    
    # Create demo users
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    db = SessionLocal()
    
    try:
        # Check if users already exist
        existing = db.query(User).first()
        if existing:
            print("⚠️  Users already exist, skipping user creation")
            return
        
        users_data = [
            {"email": "pm1@example.com", "password": "password", "role": "pm", "full_name": "Project Manager"},
            {"email": "tester1@example.com", "password": "password", "role": "tester", "full_name": "Tester One"},
            {"email": "dev1@example.com", "password": "password", "role": "developer", "full_name": "Developer One"},
        ]
        
        for user_data in users_data:
            hashed_password = pwd_context.hash(user_data["password"])
            user = User(
                email=user_data["email"],
                hashed_password=hashed_password,
                role=user_data["role"],
                full_name=user_data["full_name"]
            )
            db.add(user)
            print(f"✅ Created user: {user_data['email']}")
        
        db.commit()
        print("\n🎉 Database initialized successfully!")
        print("\nDemo Login Credentials:")
        print("PM:        pm1@example.com / password")
        print("Tester:    tester1@example.com / password")
        print("Developer: dev1@example.com / password")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_database()
