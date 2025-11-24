# routers/prompts.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from uuid import uuid4
from datetime import datetime

router = APIRouter()

class PromptCreate(BaseModel):
    name: str
    content: str
    category: str
    description: str

class PromptUpdate(BaseModel):
    name: str | None = None
    content: str | None = None
    category: str | None = None
    description: str | None = None

class Prompt(BaseModel):
    id: str
    name: str
    content: str
    category: str
    description: str
    created_at: str

PROMPTS: Dict[str, Prompt] = {}

def _seed():
    if PROMPTS:
        return
    pid = str(uuid4())[:8]
    PROMPTS[pid] = Prompt(
        id=pid,
        name="Classification Prompt",
        content="Classify the following text into categories...",
        category="classification",
        description="Generic classification prompt",
        created_at=datetime.utcnow().isoformat(),
    )

_seed()

@router.get("", response_model=List[Prompt])
async def list_prompts():
    """
    Frontend: api.getPrompts()
    """
    return list(PROMPTS.values())

@router.post("", response_model=Prompt)
async def create_prompt(body: PromptCreate):
    """
    Frontend: api.createPrompt(newPrompt)
    """
    pid = str(uuid4())[:8]
    prompt = Prompt(
        id=pid,
        name=body.name,
        content=body.content,
        category=body.category,
        description=body.description,
        created_at=datetime.utcnow().isoformat(),
    )
    PROMPTS[pid] = prompt
    return prompt

@router.delete("/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """
    Frontend: api.deletePrompt(id)
    """
    if prompt_id not in PROMPTS:
        raise HTTPException(status_code=404, detail="Prompt not found")
    del PROMPTS[prompt_id]
    return {"status": "deleted", "prompt_id": prompt_id}

@router.patch("/{prompt_id}", response_model=Prompt)
async def update_prompt(prompt_id: str, body: PromptUpdate):
    """
    Frontend: api.updatePrompt(id, updates)
    """
    prompt = PROMPTS.get(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    data = prompt.dict()
    for field in ["name", "content", "category", "description"]:
        value = getattr(body, field)
        if value is not None:
            data[field] = value

    updated = Prompt(**data)
    PROMPTS[prompt_id] = updated
    return updated
