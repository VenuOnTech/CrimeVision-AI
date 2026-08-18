from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

router = APIRouter()

@router.post("/upload")
async def upload_evidence(
    case_id: str = Form(...),
    evidence_type: str = Form(...), # "IMAGE", "VIDEO", "DOCUMENT"
    file: UploadFile = File(...)
):
    """
    Receives raw evidence files from the React frontend or Flutter app.
    Saves file to disk/cloud and logs metadata to MongoDB.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
        
    # File saving logic goes here...
    file_path = f"storage/{case_id}/{file.filename}"

    return {
        "message": "Evidence uploaded successfully",
        "evidence_id": "EVID-999",
        "case_id": case_id,
        "filename": file.filename,
        "status": "PENDING_ANALYSIS"
    }