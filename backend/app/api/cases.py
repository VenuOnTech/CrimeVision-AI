from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List
from datetime import datetime

router = APIRouter()

class CaseCreate(BaseModel):
    title: str
    description: str
    location: str
    investigator_id: str

class CaseResponse(CaseCreate):
    id: str
    created_at: datetime
    status: str

@router.post("/", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(case: CaseCreate):
    """
    Creates a new case in the MongoDB database.
    """
    # NOTE: MongoDB insertion logic will go here
    # db_manager.db_mongo["cases"].insert_one(case_dict)
    
    return {
        "id": "CASE-2026-001",
        "title": case.title,
        "description": case.description,
        "location": case.location,
        "investigator_id": case.investigator_id,
        "created_at": datetime.utcnow(),
        "status": "OPEN"
    }

@router.get("/", response_model=List[CaseResponse])
async def list_cases():
    """
    Retrieves all open cases for the web dashboard.
    """
    return []