
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from backend.database import SessionLocal, User
from backend.api.auth import get_password_hash

def create_admin():
    db = SessionLocal()
    email = "admin@kavach.ai"
    password = "admin"
    
    user = db.query(User).filter(User.email == email).first()
    if user:
        print(f"User {email} already exists.")
        return

    hashed = get_password_hash(password)
    new_user = User(email=email, hashed_password=hashed, role="admin")
    db.add(new_user)
    db.commit()
    print(f"Created admin user: {email} / {password}")
    db.close()

if __name__ == "__main__":
    create_admin()
