from fastapi import FastAPI
from contextlib import asynccontextmanager

from .database import init_db
from .iLogs.iLogs_api import router as ilogs_router
from .iNcidents.iNcidents_api import router as incidents_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database (create tables if they don't exist)
    await init_db()
    yield
    # Shutdown: add cleanup here later if needed


app = FastAPI(
    title="iThink Backend",
    description="Backend for iThink – EchoSphere 2026",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(ilogs_router, prefix="/api/v1", tags=["iLogs"])
app.include_router(incidents_router, prefix="/api/v1", tags=["iNcidents"])


@app.get("/health")
async def health():
    return {"status": "ok"}