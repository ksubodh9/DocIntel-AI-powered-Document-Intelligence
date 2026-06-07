"""
Prompt templates for all LLM tasks.
Keeping prompts centralized makes them easy to tune and version.
"""

# ── Classification ────────────────────────────────────────────────────────────

CLASSIFICATION_PROMPT = """You are a document classification expert.

Analyze the following document text (first 3000 characters) and classify it.

DOCUMENT TEXT:
{text}

Classify this document into ONE of the following categories:
- Resume
- Research Paper
- Legal Document
- Medical Report
- Invoice
- Financial Report
- User Manual
- Policy Document
- News Article
- Academic Thesis
- Business Proposal
- Contract
- Other

Respond ONLY with valid JSON in this exact format:
{{
  "document_type": "<category>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explanation>"
}}"""


# ── Summary ───────────────────────────────────────────────────────────────────

SHORT_SUMMARY_PROMPT = """You are a document summarization expert.

Summarize the following document in 2-5 concise sentences. Focus on the main purpose and most critical information.

DOCUMENT:
{text}

Write only the summary paragraph. No headers, no bullet points."""


DETAILED_SUMMARY_PROMPT = """You are a document analysis expert.

Provide a detailed analysis of the following document covering:
1. Main purpose
2. Key findings or content
3. Important details or data points
4. Conclusions or outcomes

DOCUMENT:
{text}

Write a structured analysis in clear prose. Be thorough but concise."""


TOPICS_EXTRACTION_PROMPT = """Extract structured information from this document.

DOCUMENT:
{text}

Respond ONLY with valid JSON:
{{
  "topics": ["topic1", "topic2", "topic3"],
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "entities": ["entity1", "entity2", "entity3"]
}}

- topics: 3-6 main themes or subject areas
- keywords: 5-10 important terms
- entities: named entities (people, organizations, places, products)"""


# ── Q&A / Chat ────────────────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """You are an expert document analyst. Your role is to answer questions about a document using ONLY the provided context chunks.

Rules:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. If the answer is not found in the context, say exactly: "I could not find this information in the document."
3. Always be concise and direct.
4. Do not make up or infer information not present in the context.
5. When relevant, reference the page numbers from the context."""

QA_USER_PROMPT = """Context from the document (with page references):

{context}

---
Conversation history:
{history}

---
Question: {question}

Answer based only on the context above:"""


# ── Suggested Questions ───────────────────────────────────────────────────────

SUGGESTED_QUESTIONS_PROMPT = """Based on this document, generate 5 insightful questions that a reader might want to ask.

DOCUMENT SUMMARY:
{summary}

DOCUMENT TYPE: {doc_type}

Generate questions that:
- Cover different aspects of the document
- Are specific and answerable from the document
- Range from factual to analytical

Respond ONLY with valid JSON:
{{
  "questions": [
    "Question 1?",
    "Question 2?",
    "Question 3?",
    "Question 4?",
    "Question 5?"
  ]
}}"""
