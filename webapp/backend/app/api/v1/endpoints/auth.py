"""
Authentication endpoints for user login, registration, and token management.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from datetime import timedelta
from typing import Optional
from loguru import logger

from app.services.auth_service import AuthService
from app.core.config import settings
from app.models.user import User

router = APIRouter()
security = HTTPBearer()


class UserRegister(BaseModel):
    """User registration request model."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password (minimum 8 characters)")
    first_name: str = Field(..., description="User's first name")
    last_name: str = Field(..., description="User's last name")
    company: Optional[str] = Field(None, description="Company name")


class UserLogin(BaseModel):
    """User login request model."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    remember_me: bool = Field(False, description="Remember user session")


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: dict = Field(..., description="User information")


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str = Field(..., description="Valid refresh token")


@router.post("/register", response_model=dict)
async def register_user(user_data: UserRegister):
    """Register a new user account."""
    try:
        logger.info(f"Registering new user: {user_data.email}")
        
        auth_service = AuthService()
        
        # Check if user already exists
        existing_user = await auth_service.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email address already registered"
            )
        
        # Create new user
        user = await auth_service.create_user(
            email=user_data.email,
            password=user_data.password,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            company=user_data.company
        )
        
        logger.info(f"User registered successfully: {user.email}")
        
        return {
            "message": "User registered successfully",
            "user_id": user.id,
            "email": user.email
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login_user(credentials: UserLogin):
    """Authenticate user and return access tokens."""
    try:
        logger.info(f"User login attempt: {credentials.email}")
        
        auth_service = AuthService()
        
        # Authenticate user
        user = await auth_service.authenticate_user(
            email=credentials.email,
            password=credentials.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Generate tokens
        access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        refresh_token_expires = timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        
        if credentials.remember_me:
            access_token_expires = timedelta(days=1)  # Extended session
        
        tokens = await auth_service.create_tokens(
            user_id=user.id,
            access_token_expires=access_token_expires,
            refresh_token_expires=refresh_token_expires
        )
        
        # Update last login
        await auth_service.update_last_login(user.id)
        
        logger.info(f"User logged in successfully: {user.email}")
        
        return TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds()),
            user={
                "id": user.id,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "company": user.company,
                "is_active": user.is_active
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=dict)
async def refresh_access_token(request: RefreshTokenRequest):
    """Refresh access token using valid refresh token."""
    try:
        auth_service = AuthService()
        
        # Validate refresh token and generate new access token
        new_tokens = await auth_service.refresh_tokens(request.refresh_token)
        
        if not new_tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return {
            "access_token": new_tokens["access_token"],
            "token_type": "bearer",
            "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Logout user and invalidate tokens."""
    try:
        auth_service = AuthService()
        
        # Extract token from Authorization header
        token = credentials.credentials
        
        # Add token to blacklist
        await auth_service.blacklist_token(token)
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=dict)
async def get_current_user_info(
    current_user: User = Depends(auth_service.get_current_user)
):
    """Get current authenticated user information."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "company": current_user.company,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at,
        "last_login": current_user.last_login
    }


@router.post("/forgot-password")
async def forgot_password(email: EmailStr):
    """Send password reset email."""
    try:
        auth_service = AuthService()
        
        # Send reset email (implement email service)
        await auth_service.send_password_reset_email(email)
        
        # Always return success to prevent email enumeration
        return {"message": "If email exists, password reset instructions have been sent"}
        
    except Exception as e:
        logger.error(f"Error in forgot password: {e}")
        return {"message": "If email exists, password reset instructions have been sent"}