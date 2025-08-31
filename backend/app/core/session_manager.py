from typing import Dict, Optional, Any
import uuid
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Session:
    """User session with data and state"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}

    def update_access(self):
        """Update last access time"""
        self.last_accessed = datetime.now()

    def set_data(self, data: pd.DataFrame, metadata: Dict[str, Any] = None):
        """Set session data"""
        self.data = data
        if metadata:
            self.metadata.update(metadata)
        self.update_access()

    def add_result(self, key: str, result: Any):
        """Add analysis result"""
        self.results[key] = result
        self.update_access()

    def add_model(self, key: str, model: Any):
        """Add trained model"""
        self.models[key] = model
        self.update_access()


class SessionManager:
    """Manage user sessions"""

    def __init__(self, ttl_minutes: int = 60):
        self._sessions: Dict[str, Session] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
        self._cleanup_task = None
        self._start_cleanup_task()

    def create_session(self) -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = Session(session_id)
        logger.info(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        session = self._sessions.get(session_id)
        if session:
            session.update_access()
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def _start_cleanup_task(self):
        """Start background cleanup task"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Periodically cleanup expired sessions"""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            await self._cleanup_expired()

    async def _cleanup_expired(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired = []

        for session_id, session in self._sessions.items():
            if now - session.last_accessed > self._ttl:
                expired.append(session_id)

        for session_id in expired:
            self.delete_session(session_id)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    async def cleanup(self):
        """Cleanup manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        self._sessions.clear()
