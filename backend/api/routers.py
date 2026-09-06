"""API routers."""

from fastapi import APIRouter

api_router = APIRouter()

@api_router.get("/projects")
async def list_projects() -> list[dict]:
    # Placeholder for application service call
    return []
