from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FinalAnswerSchema(BaseModel):
    answer: str = Field(..., description="Concise answer to the user's question.")
    key_points: List[str] = Field(default_factory=list, description="Supporting points.")
    data_used: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured data used to form the answer.",
    )
