"""
Authentication service for user management and JWT token handling.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from loguru import logger

from app.core.database import SessionLocal
from app.core.auth import verify_password, get_password_hash, create_access_token, create_refresh_token
from app.models.user import User


class AuthService:
    """Service for authentication operations."""
    
    def __init__(self):
        self.db = SessionLocal()
    
    def __del__(self):
        if hasattr(self, 'db'):
            self.db.close()
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        try:
            user = self.db.query(User).filter(User.email == email).first()
            return user
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            return user
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    async def create_user(
        self,
        email: str,
        password: str,
        first_name: str,
        last_name: str,
        company: Optional[str] = None
    ) -> User:
        """Create a new user account."""
        try:
            # Hash password
            hashed_password = get_password_hash(password)
            
            # Create user
            user = User(
                email=email,
                hashed_password=hashed_password,
                first_name=first_name,
                last_name=last_name,
                company=company,
                is_active=True
            )
            
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            
            logger.info(f"Created user account: {email}")
            return user
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating user: {e}")
            raise
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        try:
            user = await self.get_user_by_email(email)
            
            if not user:
                return None
            
            if not verify_password(password, user.hashed_password):
                return None
            
            return user
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    async def create_tokens(
        self,
        user_id: int,
        access_token_expires: Optional[timedelta] = None,
        refresh_token_expires: Optional[timedelta] = None
    ) -> Dict[str, str]:
        """Create access and refresh tokens for user."""
        try:
            # Create token data
            token_data = {"sub": str(user_id)}
            
            # Create tokens
            access_token = create_access_token(
                data=token_data,
                expires_delta=access_token_expires
            )
            
            refresh_token = create_refresh_token(
                data=token_data,
                expires_delta=refresh_token_expires
            )
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token
            }
            
        except Exception as e:
            logger.error(f"Error creating tokens: {e}")
            raise
    
    async def refresh_tokens(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """Refresh access token using refresh token."""
        try:
            from app.core.auth import verify_token
            
            # Verify refresh token
            payload = verify_token(refresh_token)
            
            # Check if it's a refresh token
            if payload.get("type") != "refresh":
                return None
            
            # Extract user ID
            user_id = payload.get("sub")
            if not user_id:
                return None
            
            # Verify user exists and is active
            user = await self.get_user_by_id(int(user_id))
            if not user or not user.is_active:
                return None
            
            # Create new tokens
            tokens = await self.create_tokens(user_id=int(user_id))
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error refreshing tokens: {e}")
            return None
    
    async def blacklist_token(self, token: str) -> bool:
        """Add token to blacklist (implement with Redis/cache in production)."""
        try:
            # For now, we'll log the logout
            # In production, implement proper token blacklisting with Redis
            logger.info(f"Token blacklisted (logout)")
            return True
            
        except Exception as e:
            logger.error(f"Error blacklisting token: {e}")
            return False
    
    async def update_last_login(self, user_id: int) -> bool:
        """Update user's last login timestamp."""
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if user:
                user.last_login = datetime.utcnow()
                self.db.commit()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating last login: {e}")
            return False
    
    async def send_password_reset_email(self, email: str) -> bool:
        """Send password reset email (placeholder for email service integration)."""
        try:
            user = await self.get_user_by_email(email)
            
            if user:
                # In production, implement actual email sending
                # Generate reset token, send email, etc.
                logger.info(f"Password reset requested for: {email}")
                pass
            
            # Always return success to prevent email enumeration
            return True
            
        except Exception as e:
            logger.error(f"Error in password reset: {e}")
            return True  # Still return True to prevent enumeration
    
    def get_current_user(self):
        """Dependency function for getting current user."""
        from app.core.auth import get_current_user
        return get_current_user