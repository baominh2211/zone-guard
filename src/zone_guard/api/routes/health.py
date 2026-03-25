"""Health check."""
from fastapi import APIRouter, Request
from zone_guard.api.schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    components = {}
    status = "healthy"
    try:
        from zone_guard.db.database import get_session
        from sqlalchemy import text
        async with get_session() as s:
            await s.execute(text("SELECT 1"))
        components["database"] = {"status": "healthy"}
    except Exception as e:
        components["database"] = {"status": "unhealthy", "error": str(e)}
        status = "degraded"
    app = getattr(request.app.state, "app_instance", None)
    if app and hasattr(app, "camera_mgr"):
        st = app.camera_mgr.get_all_status() if hasattr(app.camera_mgr, "get_all_status") else []
        components["cameras"] = {"online": sum(1 for s in st if s.get("connected")), "total": len(st)}
    return HealthResponse(status=status, components=components, version="1.0.0")
