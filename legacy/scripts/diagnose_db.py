
import asyncio
from sqlalchemy import select
from backend.database import SessionLocal, User
from passlib.context import CryptContext

async def verify_user():
    pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
    async with SessionLocal() as db:
        result = await db.execute(select(User).filter(User.email == "demo@kavach.ai"))
        user = result.scalars().first()
        if not user:
            print("User demo@kavach.ai NOT found!")
            return
        
        print(f"User found: {user.email}")
        print(f"Role: {user.role}")
        
        test_pass = "kavach2026"
        is_correct = pwd_context.verify(test_pass, user.hashed_password)
        print(f"Password 'kavach2026' is {'CORRECT' if is_correct else 'INCORRECT'}")

if __name__ == "__main__":
    asyncio.run(verify_user())
