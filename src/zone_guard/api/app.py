"""FastAPI application factory."""
import os, logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from zone_guard.api.routes import auth, events, health, stream

logger = logging.getLogger(__name__)

def create_api(app_instance=None):
    @asynccontextmanager
    async def lifespan(api):
        yield
    api = FastAPI(title="ZoneGuard API", version="1.0.0", lifespan=lifespan)
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    api.state.app_instance = app_instance
    os.makedirs("data/snapshots", exist_ok=True)
    api.mount("/static/snapshots", StaticFiles(directory="data/snapshots"), name="snapshots")
    api.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
    api.include_router(events.router, prefix="/api/events", tags=["Events"])
    api.include_router(stream.router, prefix="/api/stream", tags=["Stream"])
    api.include_router(health.router, tags=["Health"])
    return api
