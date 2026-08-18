from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import db_manager
from app.api import auth, cases, evidence, analysis

# Manage Database Startup and Shutdown lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_manager.connect_dbs()
    yield
    await db_manager.close_dbs()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Set up Cross-Origin Resource Sharing (CORS) for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["Authentication"])
app.include_router(cases.router, prefix=f"{settings.API_V1_STR}/cases", tags=["Case Management"])
app.include_router(evidence.router, prefix=f"{settings.API_V1_STR}/evidence", tags=["Evidence Upload"])
app.include_router(analysis.router, prefix=f"{settings.API_V1_STR}/analysis", tags=["Research Engine Analysis"])

@app.get("/")
async def root():
    return {"message": "CrimeVision AI Backend API is running."}