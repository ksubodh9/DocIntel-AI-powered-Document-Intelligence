-- =============================================================================
-- PDF Agent — Supabase SQL Schema
-- Run this in: Supabase Dashboard → SQL Editor → New Query → Run
-- =============================================================================

-- Enable UUID extension (already on by default in Supabase, but just in case)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- documents
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.documents (
    id                       UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id                  UUID        REFERENCES auth.users(id) ON DELETE SET NULL,

    filename                 TEXT        NOT NULL,
    original_filename        TEXT        NOT NULL,
    file_path                TEXT        NOT NULL,
    file_size                INTEGER     NOT NULL,
    page_count               INTEGER     DEFAULT 0,

    -- Cached extracted text (avoids re-parsing on every request)
    full_text                TEXT,

    -- Classification
    document_type            TEXT,
    classification_confidence REAL,

    -- Processing state: uploaded | processing | ready | error
    status                   TEXT        NOT NULL DEFAULT 'uploaded',
    error_message            TEXT,

    -- AI-generated content
    short_summary            TEXT,
    detailed_summary         TEXT,
    topics                   JSONB,          -- TEXT[]
    keywords                 JSONB,          -- TEXT[]
    entities                 JSONB,          -- TEXT[]
    suggested_questions      JSONB,          -- TEXT[]

    -- Metadata (native PDF headers + LLM-extracted)
    doc_metadata             JSONB,

    -- Table extraction
    has_tables               BOOLEAN     DEFAULT FALSE,
    table_count              INTEGER     DEFAULT 0,
    tables                   JSONB,          -- [{page, markdown, caption}]

    -- ChromaDB collection
    collection_name          TEXT,

    created_at               TIMESTAMPTZ DEFAULT NOW(),
    updated_at               TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_user_id   ON public.documents (user_id);
CREATE INDEX IF NOT EXISTS idx_documents_status    ON public.documents (status);
CREATE INDEX IF NOT EXISTS idx_documents_created   ON public.documents (created_at DESC);

-- auto-update updated_at
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_documents_updated_at ON public.documents;
CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON public.documents
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- =============================================================================
-- chat_messages
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.chat_messages (
    id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID        NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    role        TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT        NOT NULL,
    citations   JSONB,          -- [{page, text, chunk_id}]
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_document_id ON public.chat_messages (document_id);

-- =============================================================================
-- usage_events
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.usage_events (
    id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID        REFERENCES auth.users(id) ON DELETE SET NULL,
    document_id UUID        REFERENCES public.documents(id) ON DELETE SET NULL,
    -- event_type: upload | classify | summarize | chat | compare | multi_chat
    event_type  TEXT        NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_usage_events_user_id    ON public.usage_events (user_id);
CREATE INDEX IF NOT EXISTS idx_usage_events_document_id ON public.usage_events (document_id);
CREATE INDEX IF NOT EXISTS idx_usage_events_created    ON public.usage_events (created_at DESC);

-- =============================================================================
-- feedback
-- General product feedback. The form collects only rating + comment; category
-- is retained (default 'general') for future triage. Written by the backend
-- (service role), so RLS below is a safety net rather than the primary path.
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.feedback (
    id                UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id           UUID        REFERENCES auth.users(id) ON DELETE SET NULL,
    email             TEXT,
    category          TEXT        NOT NULL DEFAULT 'general',  -- general | bug | idea
    rating            INTEGER     CHECK (rating BETWEEN 1 AND 5),
    comment           TEXT,
    route             TEXT,
    last_feature_used TEXT,
    user_agent        TEXT,
    status            TEXT        NOT NULL DEFAULT 'new',       -- new | reviewed
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON public.feedback (user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON public.feedback (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_status  ON public.feedback (status);

-- =============================================================================
-- Row Level Security (RLS)
-- Users can only read/write their own rows.
-- Service role (backend) bypasses RLS automatically.
-- =============================================================================

ALTER TABLE public.documents     ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.usage_events  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.feedback      ENABLE ROW LEVEL SECURITY;

-- Documents: owner full access
CREATE POLICY "docs_owner_select" ON public.documents
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "docs_owner_insert" ON public.documents
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "docs_owner_update" ON public.documents
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "docs_owner_delete" ON public.documents
    FOR DELETE USING (auth.uid() = user_id);

-- Chat messages: owner access via document ownership
CREATE POLICY "chat_owner_select" ON public.chat_messages
    FOR SELECT USING (
        document_id IN (
            SELECT id FROM public.documents WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "chat_owner_insert" ON public.chat_messages
    FOR INSERT WITH CHECK (
        document_id IN (
            SELECT id FROM public.documents WHERE user_id = auth.uid()
        )
    );

-- Usage events: users see only their own
CREATE POLICY "usage_owner_select" ON public.usage_events
    FOR SELECT USING (auth.uid() = user_id);

-- Feedback: users may insert their own rows and read them back. Admin/service
-- role reads everything (bypasses RLS). Anonymous submissions go via the backend.
CREATE POLICY "feedback_owner_insert" ON public.feedback
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "feedback_owner_select" ON public.feedback
    FOR SELECT USING (auth.uid() = user_id);

-- =============================================================================
-- Admin role helper
-- Call this from the Supabase SQL editor to grant admin to a user:
--   SELECT grant_admin('user-uuid-here');
-- =============================================================================
CREATE OR REPLACE FUNCTION public.grant_admin(target_user_id UUID)
RETURNS VOID LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
    UPDATE auth.users
    SET    raw_app_meta_data = raw_app_meta_data || '{"role": "admin"}'::jsonb
    WHERE  id = target_user_id;
END;
$$;
