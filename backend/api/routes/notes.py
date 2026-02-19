"""User notes API endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from backend.database.db import get_db
from backend.database.models import UserNote

router = APIRouter(prefix="/notes", tags=["notes"])


class NoteCreate(BaseModel):
    title: Optional[str] = None
    content: str
    symbol: Optional[str] = None
    prediction_id: Optional[int] = None


class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None


@router.post("")
def create_note(note: NoteCreate, db: Session = Depends(get_db)):
    """Create a new note."""
    db_note = UserNote(
        title=note.title,
        content=note.content,
        symbol=note.symbol,
        prediction_id=note.prediction_id
    )
    db.add(db_note)
    db.commit()
    db.refresh(db_note)
    
    return {
        "id": db_note.id,
        "title": db_note.title,
        "content": db_note.content,
        "symbol": db_note.symbol,
        "prediction_id": db_note.prediction_id,
        "created_at": db_note.created_at.isoformat(),
        "updated_at": db_note.updated_at.isoformat()
    }


@router.get("")
def get_notes(
    symbol: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> List[dict]:
    """Get user notes."""
    query = db.query(UserNote)
    
    if symbol:
        query = query.filter(UserNote.symbol == symbol)
    
    notes = query.order_by(UserNote.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": note.id,
            "title": note.title,
            "content": note.content,
            "symbol": note.symbol,
            "prediction_id": note.prediction_id,
            "created_at": note.created_at.isoformat(),
            "updated_at": note.updated_at.isoformat()
        }
        for note in notes
    ]


@router.get("/{note_id}")
def get_note(note_id: int, db: Session = Depends(get_db)):
    """Get a specific note by ID."""
    note = db.query(UserNote).filter(UserNote.id == note_id).first()
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    return {
        "id": note.id,
        "title": note.title,
        "content": note.content,
        "symbol": note.symbol,
        "prediction_id": note.prediction_id,
        "created_at": note.created_at.isoformat(),
        "updated_at": note.updated_at.isoformat()
    }


@router.put("/{note_id}")
def update_note(note_id: int, note_update: NoteUpdate, db: Session = Depends(get_db)):
    """Update a note."""
    note = db.query(UserNote).filter(UserNote.id == note_id).first()
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    if note_update.title is not None:
        note.title = note_update.title
    if note_update.content is not None:
        note.content = note_update.content
    
    note.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(note)
    
    return {
        "id": note.id,
        "title": note.title,
        "content": note.content,
        "symbol": note.symbol,
        "prediction_id": note.prediction_id,
        "created_at": note.created_at.isoformat(),
        "updated_at": note.updated_at.isoformat()
    }


@router.delete("/{note_id}")
def delete_note(note_id: int, db: Session = Depends(get_db)):
    """Delete a note."""
    note = db.query(UserNote).filter(UserNote.id == note_id).first()
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    db.delete(note)
    db.commit()
    
    return {"message": "Note deleted successfully"}
