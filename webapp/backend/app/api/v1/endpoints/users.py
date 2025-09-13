"""
User management and profile endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field
from loguru import logger

from app.services.user_service import UserService
from app.core.auth import get_current_user
from app.models.user import User

router = APIRouter()


class UserProfileUpdate(BaseModel):
    """User profile update model."""
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    company: Optional[str] = Field(None, description="Company name")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")


class PasswordChange(BaseModel):
    """Password change model."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password (minimum 8 characters)")


class UserPreferences(BaseModel):
    """User preferences model."""
    default_optimization_method: str = Field("max_sharpe", description="Default optimization method")
    default_regime_method: str = Field("investment_clock", description="Default regime method")
    risk_tolerance: str = Field("moderate", description="Risk tolerance level")
    notification_settings: Dict[str, bool] = Field(default_factory=dict, description="Notification preferences")
    dashboard_layout: Dict[str, Any] = Field(default_factory=dict, description="Dashboard configuration")
    theme: str = Field("light", description="UI theme preference")


@router.get("/profile", response_model=Dict[str, Any])
async def get_user_profile(
    current_user: User = Depends(get_current_user)
):
    """Get current user profile."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "company": current_user.company,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at,
        "last_login": current_user.last_login,
        "preferences": current_user.preferences or {}
    }


@router.put("/profile", response_model=Dict[str, Any])
async def update_user_profile(
    profile_data: UserProfileUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update user profile information."""
    try:
        logger.info(f"Updating profile for user {current_user.id}")
        
        user_service = UserService()
        
        # Update user profile
        updated_user = await user_service.update_user_profile(
            user_id=current_user.id,
            first_name=profile_data.first_name,
            last_name=profile_data.last_name,
            company=profile_data.company,
            preferences=profile_data.preferences
        )
        
        return {
            "message": "Profile updated successfully",
            "user": {
                "id": updated_user.id,
                "email": updated_user.email,
                "first_name": updated_user.first_name,
                "last_name": updated_user.last_name,
                "company": updated_user.company,
                "preferences": updated_user.preferences
            }
        }
        
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user)
):
    """Change user password."""
    try:
        logger.info(f"Password change request for user {current_user.id}")
        
        user_service = UserService()
        
        # Change password
        success = await user_service.change_password(
            user_id=current_user.id,
            current_password=password_data.current_password,
            new_password=password_data.new_password
        )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Current password is incorrect"
            )
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(status_code=500, detail="Password change failed")


@router.get("/preferences", response_model=UserPreferences)
async def get_user_preferences(
    current_user: User = Depends(get_current_user)
):
    """Get user preferences."""
    preferences = current_user.preferences or {}
    
    return UserPreferences(
        default_optimization_method=preferences.get("default_optimization_method", "max_sharpe"),
        default_regime_method=preferences.get("default_regime_method", "investment_clock"),
        risk_tolerance=preferences.get("risk_tolerance", "moderate"),
        notification_settings=preferences.get("notification_settings", {}),
        dashboard_layout=preferences.get("dashboard_layout", {}),
        theme=preferences.get("theme", "light")
    )


@router.put("/preferences", response_model=Dict[str, Any])
async def update_user_preferences(
    preferences: UserPreferences,
    current_user: User = Depends(get_current_user)
):
    """Update user preferences."""
    try:
        logger.info(f"Updating preferences for user {current_user.id}")
        
        user_service = UserService()
        
        # Update preferences
        updated_user = await user_service.update_user_preferences(
            user_id=current_user.id,
            preferences=preferences.dict()
        )
        
        return {
            "message": "Preferences updated successfully",
            "preferences": updated_user.preferences
        }
        
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activity", response_model=List[Dict[str, Any]])
async def get_user_activity(
    current_user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0
):
    """Get user activity history."""
    try:
        user_service = UserService()
        
        activities = await user_service.get_user_activity(
            user_id=current_user.id,
            limit=limit,
            offset=offset
        )
        
        return [
            {
                "id": activity.id,
                "action": activity.action,
                "resource_type": activity.resource_type,
                "resource_id": activity.resource_id,
                "timestamp": activity.timestamp,
                "details": activity.details
            }
            for activity in activities
        ]
        
    except Exception as e:
        logger.error(f"Error getting user activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/usage-stats", response_model=Dict[str, Any])
async def get_usage_statistics(
    current_user: User = Depends(get_current_user)
):
    """Get user usage statistics."""
    try:
        user_service = UserService()
        
        stats = await user_service.get_usage_statistics(user_id=current_user.id)
        
        return {
            "portfolios_created": stats["portfolios_count"],
            "backtests_run": stats["backtests_count"],
            "scenarios_analyzed": stats["scenarios_count"],
            "total_api_calls": stats["api_calls_count"],
            "account_age_days": stats["account_age"],
            "last_activity": stats["last_activity"],
            "favorite_strategies": stats["favorite_strategies"],
            "preferred_assets": stats["preferred_assets"]
        }
        
    except Exception as e:
        logger.error(f"Error getting usage statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export-data")
async def export_user_data(
    current_user: User = Depends(get_current_user)
):
    """Export user data for download."""
    try:
        logger.info(f"Data export request for user {current_user.id}")
        
        user_service = UserService()
        
        # Generate export file
        export_data = await user_service.export_user_data(user_id=current_user.id)
        
        return {
            "export_id": export_data["export_id"],
            "status": "processing",
            "download_url": f"/users/download/{export_data['export_id']}",
            "estimated_completion": "2-5 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error exporting user data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/account")
async def delete_user_account(
    current_user: User = Depends(get_current_user),
    confirmation: str = Field(..., description="Type 'DELETE' to confirm")
):
    """Delete user account and all associated data."""
    if confirmation != "DELETE":
        raise HTTPException(
            status_code=400,
            detail="Account deletion requires typing 'DELETE' as confirmation"
        )
    
    try:
        logger.warning(f"Account deletion request for user {current_user.id}")
        
        user_service = UserService()
        
        # Schedule account deletion
        deletion_task = await user_service.schedule_account_deletion(
            user_id=current_user.id
        )
        
        return {
            "message": "Account deletion scheduled",
            "deletion_date": deletion_task["deletion_date"],
            "cancellation_deadline": deletion_task["cancellation_deadline"]
        }
        
    except Exception as e:
        logger.error(f"Error scheduling account deletion: {e}")
        raise HTTPException(status_code=500, detail="Account deletion failed")