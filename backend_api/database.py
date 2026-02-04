"""
Database connection and session management using SQLAlchemy.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get DATABASE_URL from environment (set in Render Dashboard)
DATABASE_URL = os.environ.get("DATABASE_URL")

# Handle Render's postgres:// vs postgresql:// issue
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine only if DATABASE_URL is available
engine = None
SessionLocal = None
Base = declarative_base()

def init_db():
    """Initialize database connection. Called on startup."""
    global engine, SessionLocal
    
    if not DATABASE_URL:
        print("WARNING: DATABASE_URL not set. History features disabled.")
        return False
    
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)
        print("Database connected successfully!")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

def get_db():
    """Dependency to get database session."""
    if SessionLocal is None:
        return None
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def is_db_available():
    """Check if database is available."""
    return engine is not None and SessionLocal is not None
