
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from backend.database import engine, Base
from backend.database.models import User, ScanResult, AuditLog

def init_db():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")
    print(f"Database URL: {engine.url}")

if __name__ == "__main__":
    init_db()
