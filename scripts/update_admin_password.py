
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from backend.database import SessionLocal, User
from backend.api.auth import get_password_hash

def update_admin_password():
    db = SessionLocal()
    email = "admin@kavach.ai"
    new_password = "Kavach@2026"  # Updated password
    
    user = db.query(User).filter(User.email == email).first()
    if not user:
        print(f"User {email} does not exist. Creating new admin...")
        hashed = get_password_hash(new_password)
        new_user = User(email=email, hashed_password=hashed, role="admin")
        db.add(new_user)
        db.commit()
        print(f"Created admin user: {email} / {new_password}")
    else:
        # Update existing user's password
        user.hashed_password = get_password_hash(new_password)
        db.commit()
        print(f"Updated admin password: {email} / {new_password}")
    
    db.close()

if __name__ == "__main__":
    update_admin_password()
