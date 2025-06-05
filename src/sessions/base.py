from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseSession(ABC):
    """Base class for session management."""
    
    @abstractmethod
    def create_session(self, session_id: str) -> None:
        """Create a new session."""
        pass
    
    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        pass
    
    @abstractmethod
    def update_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Update session data."""
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        pass
    
    @abstractmethod
    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all sessions."""
        pass 