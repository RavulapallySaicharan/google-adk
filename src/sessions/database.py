from typing import Any, Dict, Optional
import json
from sqlalchemy import create_engine, Column, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .base import BaseSession

Base = declarative_base()

class SessionModel(Base):
    """SQLAlchemy model for sessions."""
    __tablename__ = 'sessions'
    
    session_id = Column(String, primary_key=True)
    data = Column(JSON)

class DatabaseSession(BaseSession):
    """Database-backed session management implementation."""
    
    def __init__(self, db_url: str = "sqlite:///sessions.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def create_session(self, session_id: str) -> None:
        """Create a new session in the database."""
        session = self.Session()
        try:
            if not session.query(SessionModel).filter_by(session_id=session_id).first():
                new_session = SessionModel(session_id=session_id, data={})
                session.add(new_session)
                session.commit()
        finally:
            session.close()
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data from the database."""
        session = self.Session()
        try:
            db_session = session.query(SessionModel).filter_by(session_id=session_id).first()
            return db_session.data if db_session else None
        finally:
            session.close()
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Update session data in the database."""
        session = self.Session()
        try:
            db_session = session.query(SessionModel).filter_by(session_id=session_id).first()
            if db_session:
                current_data = db_session.data or {}
                current_data.update(data)
                db_session.data = current_data
                session.commit()
        finally:
            session.close()
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session from the database."""
        session = self.Session()
        try:
            db_session = session.query(SessionModel).filter_by(session_id=session_id).first()
            if db_session:
                session.delete(db_session)
                session.commit()
        finally:
            session.close()
    
    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all sessions from the database."""
        session = self.Session()
        try:
            db_sessions = session.query(SessionModel).all()
            return {s.session_id: s.data for s in db_sessions}
        finally:
            session.close() 