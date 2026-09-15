from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from .db import db_cursor
from .dependencies import get_current_user
from .models import User

router = APIRouter(prefix="/api/auth", tags=["auth"])


class ProfileResponse(BaseModel):
    success: bool
    user: dict


class UsersListResponse(BaseModel):
    success: bool
    users: list[dict]


@router.get("/me", response_model=ProfileResponse)
async def me(user: User = Depends(get_current_user)):
    """
    Devuelve el perfil local sincronizado con Supabase.
    Si el usuario no existe localmente, se crea automáticamente.
    """
    return ProfileResponse(success=True, user=user.model_dump())


@router.get("/users", response_model=UsersListResponse)
async def list_users(user: User = Depends(get_current_user)):
    """
    Lista los usuarios registrados en app_auth.users.
    Solo disponible para administradores (role = "admin").
    """
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Solo administradores",
        )

    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT id, email, full_name, role, is_active, email_verified, created_at
            FROM app_auth.users
            ORDER BY created_at DESC
            LIMIT 200
            """
        )
        rows = cursor.fetchall()

    return UsersListResponse(success=True, users=[dict(r) for r in rows])


@router.post("/logout")
async def logout():
    """
    Con Supabase Auth los tokens son stateless.
    El logout real se hace en el frontend con supabase.auth.signOut().
    """
    return {"success": True, "message": "Sesión cerrada"}
