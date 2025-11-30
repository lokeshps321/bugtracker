#!/usr/bin/env python3
"""
Database migration script to add 'title' column to existing bugs table
without losing data.
"""
import sqlite3
import os

DB_PATH = "bugflow.db"

def migrate_database():
    """Add title column to bugs table if it doesn't exist"""
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Database file '{DB_PATH}' not found!")
        print("Creating new database with init_users.py...")
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if title column already exists
        cursor.execute("PRAGMA table_info(bugs)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'title' in columns:
            print("✅ 'title' column already exists in bugs table")
            conn.close()
            return True
        
        print("📝 Adding 'title' column to bugs table...")
        
        # Add title column (SQLite ALTER TABLE ADD COLUMN)
        cursor.execute("""
            ALTER TABLE bugs 
            ADD COLUMN title TEXT
        """)
        
        # Update existing bugs with auto-generated titles from description
        cursor.execute("""
            UPDATE bugs 
            SET title = SUBSTR(description, 1, 50) || '...'
            WHERE title IS NULL
        """)
        
        conn.commit()
        
        # Verify the migration
        cursor.execute("SELECT COUNT(*) FROM bugs")
        bug_count = cursor.fetchone()[0]
        
        print(f"✅ Migration successful! {bug_count} existing bugs preserved.")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("BugFlow Database Migration - Adding Title Column")
    print("=" * 60)
    
    success = migrate_database()
    
    if success:
        print("\n✅ Database migration completed successfully!")
        print("All your previous bug reports are intact.")
    else:
        print("\n⚠️ Migration could not be completed.")
        print("Running init_users.py to create fresh database...")
        os.system("./venv/bin/python init_users.py")
