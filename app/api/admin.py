"""
Admin API endpoints.
All require is_admin = True in the JWT app_metadata.

GET /admin/stats              — system-wide health metrics
GET /admin/users              — list of all users with usage stats
GET /admin/users/{user_id}/documents — documents for one user
"""

import logging
import os
from typing import Optional

import requests as http_requests
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.middleware.auth import require_admin
from app.models.document import Document
from app.models.usage import UsageEvent
from app.models.feedback import Feedback
from app.schemas.feedback import AdminFeedbackItem
from app.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/admin", tags=["Admin"])


def _get_supabase_users() -> list[dict]:
    """
    Fetch all users from Supabase Auth API using the service role key.
    Returns a list of {id, email, created_at, last_sign_in_at}.
    Falls back to empty list if not configured.
    """
    url = settings.supabase_url
    key = settings.supabase_service_role_key
    if not url or not key:
        logger.warning("[Admin] SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set — user list will be empty.")
        return []
    try:
        resp = http_requests.get(
            f"{url}/auth/v1/admin/users",
            headers={"apikey": key, "Authorization": f"Bearer {key}"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("users", [])
    except Exception as e:
        logger.error(f"[Admin] Failed to fetch Supabase users: {e}")
        return []


@router.get("/stats")
def get_stats(
    db: Session = Depends(get_db),
    _admin: dict = Depends(require_admin),
):
    """System health: total users, documents, API calls, storage."""
    total_docs = db.query(func.count(Document.id)).scalar() or 0
    docs_indexed = db.query(func.count(Document.id)).filter(Document.status == "ready").scalar() or 0
    total_calls = db.query(func.count(UsageEvent.id)).scalar() or 0

    # Storage: sum file sizes
    storage_bytes = db.query(func.sum(Document.file_size)).scalar() or 0

    # Unique users
    total_users = db.query(func.count(func.distinct(Document.user_id))).scalar() or 0

    # Feedback metrics
    total_feedback = db.query(func.count(Feedback.id)).scalar() or 0
    new_feedback = db.query(func.count(Feedback.id)).filter(Feedback.status == "new").scalar() or 0
    avg_rating = db.query(func.avg(Feedback.rating)).filter(Feedback.rating.isnot(None)).scalar()

    return {
        "total_users": total_users,
        "total_documents": total_docs,
        "docs_indexed": docs_indexed,
        "total_api_calls": total_calls,
        "storage_used_bytes": int(storage_bytes),
        "total_feedback": total_feedback,
        "new_feedback": new_feedback,
        "avg_rating": round(float(avg_rating), 2) if avg_rating is not None else None,
    }


@router.get("/feedback", response_model=list[AdminFeedbackItem])
def list_feedback(
    db: Session = Depends(get_db),
    _admin: dict = Depends(require_admin),
):
    """Most recent product feedback, newest first."""
    rows = (
        db.query(Feedback)
        .order_by(Feedback.created_at.desc())
        .limit(200)
        .all()
    )
    return [AdminFeedbackItem.model_validate(r) for r in rows]


@router.get("/users")
def list_users(
    db: Session = Depends(get_db),
    _admin: dict = Depends(require_admin),
):
    """All users with document and API call counts."""
    supabase_users = _get_supabase_users()
    su_map = {u["id"]: u for u in supabase_users}

    # Aggregate per user_id from our DB
    doc_counts = dict(
        db.query(Document.user_id, func.count(Document.id))
        .filter(Document.user_id.isnot(None))
        .group_by(Document.user_id)
        .all()
    )
    call_counts = dict(
        db.query(UsageEvent.user_id, func.count(UsageEvent.id))
        .filter(UsageEvent.user_id.isnot(None))
        .group_by(UsageEvent.user_id)
        .all()
    )
    last_active = dict(
        db.query(UsageEvent.user_id, func.max(UsageEvent.created_at))
        .filter(UsageEvent.user_id.isnot(None))
        .group_by(UsageEvent.user_id)
        .all()
    )

    # Merge Supabase users with DB stats
    all_user_ids = set(doc_counts.keys()) | set(call_counts.keys()) | set(su_map.keys())
    result = []
    for uid in all_user_ids:
        su = su_map.get(uid, {})
        result.append({
            "user_id": uid,
            "email": su.get("email", "unknown@unknown.com"),
            "joined_at": su.get("created_at"),
            "last_active": last_active.get(uid),
            "documents_count": doc_counts.get(uid, 0),
            "api_calls_count": call_counts.get(uid, 0),
        })

    result.sort(key=lambda u: u["api_calls_count"], reverse=True)
    return result


@router.get("/users/{user_id}/documents")
def get_user_documents(
    user_id: str,
    db: Session = Depends(get_db),
    _admin: dict = Depends(require_admin),
):
    """Documents belonging to a specific user."""
    docs = (
        db.query(Document)
        .filter(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
        .limit(50)
        .all()
    )
    from app.schemas.document import DocumentDetail
    return [DocumentDetail.model_validate(d) for d in docs]
