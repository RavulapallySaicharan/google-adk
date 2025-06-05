from typing import Any, Dict, Optional
from .base import BaseSession

class InMemorySession(BaseSession):
    """In-memory session management implementation."""
    
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, session_id: str) -> None:
        """Create a new session in memory."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {}
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data from memory."""
        return self._sessions.get(session_id)
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Update session data in memory."""
        if session_id in self._sessions:
            self._sessions[session_id].update(data)
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session from memory."""
        self._sessions.pop(session_id, None)
    
    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all sessions in memory."""
        return self._sessions.copy() 