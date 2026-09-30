"""
Scholarship & Exam Information Agent - core logic.

Capabilities:

1. RAG Q&A
   college_info.txt
        ↓
   chunks
        ↓
   TF-IDF retrieval
        ↓
   relevant context
        ↓
   Groq GPT-OSS-20B

2. Structured eligibility checking
   college_info.txt
        ↓
   structured records
        ↓
   Python rule checking

3. Deadline and exam-date tracking
   college_info.txt
        ↓
   structured dates
        ↓
   Python date filtering

Works offline when no Groq API key is available.
"""

from __future__ import annotations

import os
import re

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# ENVIRONMENT
# =========================================================

try:

    from dotenv import load_dotenv

    load_dotenv()

except ImportError:

    pass


# =========================================================
# GROQ
# =========================================================

try:

    from groq import Groq

    GROQ_AVAILABLE = True

except ImportError:

    GROQ_AVAILABLE = False


# =========================================================
# CONFIGURATION
# =========================================================

DEFAULT_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-20b"
)

CATEGORIES = [
    "General",
    "OBC",
    "SC",
    "ST",
    "EWS"
]

FIELD_RE = re.compile(
    r"^([A-Za-z][A-Za-z ]*):\s*(.+)$"
)


# =========================================================
# RECORD
# =========================================================

@dataclass
class Record:

    name: str

    raw: str

    fields: dict = field(
        default_factory=dict
    )

    @property
    def kind(self) -> str:

        return self.fields.get(
            "type",
            ""
        ).lower()


    def _date(
        self,
        key: str
    ):

        try:

            return date.fromisoformat(
                self.fields[key]
            )

        except (
            KeyError,
            ValueError
        ):

            return None


    def _num(
        self,
        key: str
    ):

        value = re.sub(
            r"[^\d.]",
            "",
            self.fields.get(
                key,
                ""
            )
        )

        try:

            return float(value)

        except ValueError:

            return None


    @property
    def deadline(self):

        return self._date(
            "deadline"
        )


    @property
    def exam_date(self):

        return self._date(
            "exam date"
        )


    @property
    def min_cgpa(self):

        return (
            self._num("min cgpa")
            or 0.0
        )


    @property
    def max_income(self):

        return self._num(
            "max family income"
        )


    @property
    def categories(self):

        raw = self.fields.get(
            "categories",
            "All"
        )

        return {
            c.strip().lower()
            for c in raw.split(",")
            if c.strip()
        }


# =========================================================
# PARSE STRUCTURED RECORDS
# =========================================================

def parse_records(
    text: str
) -> list[Record]:

    records = []

    for block in re.split(
        r"\n\s*\n",
        text
    ):

        lines = (
            block
            .strip()
            .splitlines()
        )

        if (
            not lines
            or not lines[0].startswith("## ")
        ):

            continue

        record = Record(
            name=lines[0][3:].strip(),
            raw=block.strip()
        )

        for line in lines[1:]:

            match = FIELD_RE.match(
                line.strip()
            )

            if match:

                key = (
                    match.group(1)
                    .strip()
                    .lower()
                )

                value = (
                    match.group(2)
                    .strip()
                )

                record.fields[key] = value

        if record.fields:

            records.append(
                record
            )

    return records


# =========================================================
# TEXT CHUNKING
# =========================================================

def chunk_text(
    text: str,
    max_words: int = 120
) -> list[str]:

    chunks = []

    for block in re.split(
        r"\n\s*\n",
        text
    ):

        block = block.strip()

        if (
            not block
            or block.startswith("# NOTE")
            or block.startswith("# Keep")
        ):

            continue

        words = block.split()

        if len(words) <= max_words:

            chunks.append(
                block
            )

        else:

            for i in range(
                0,
                len(words),
                max_words
            ):

                chunks.append(
                    " ".join(
                        words[
                            i:i + max_words
                        ]
                    )
                )

    return chunks


# =========================================================
# TF-IDF RETRIEVER
# =========================================================

class Retriever:

    def __init__(
        self,
        chunks: list[str]
    ):

        if not chunks:

            raise ValueError(
                "No text chunks to index."
            )

        self.chunks = chunks

        self.vectorizer = (
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2)
            )
        )

        self.matrix = (
            self.vectorizer.fit_transform(
                chunks
            )
        )


    def search(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.05
    ):

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        scores = (
            cosine_similarity(
                query_vector,
                self.matrix
            )
            .ravel()
        )

        top_indices = (
            scores.argsort()[::-1][:k]
        )

        return [
            (
                self.chunks[i],
                float(scores[i])
            )

            for i in top_indices

            if scores[i] >= min_score
        ]


# =========================================================
# SCHOLARSHIP AGENT
# =========================================================

class ScholarshipAgent:

    def __init__(
        self,
        info_path: str = "college_info.txt",
        api_key: str | None = None,
        model: str = DEFAULT_MODEL
    ):

        # ---------------------------------------------
        # Load knowledge base
        # ---------------------------------------------

        text = Path(
            info_path
        ).read_text(
            encoding="utf-8"
        )

        # ---------------------------------------------
        # Structured records
        # ---------------------------------------------

        self.records = parse_records(
            text
        )

        # ---------------------------------------------
        # RAG retriever
        # ---------------------------------------------

        self.retriever = Retriever(
            chunk_text(text)
        )

        # ---------------------------------------------
        # Groq configuration
        # ---------------------------------------------

        self.model = model

        self.client = None

        key = (
            api_key
            or os.getenv(
                "GROQ_API_KEY"
            )
        )

        if (
            key
            and GROQ_AVAILABLE
        ):

            self.client = Groq(
                api_key=key
            )


    # =====================================================
    # ONLINE STATUS
    # =====================================================

    @property
    def online(self) -> bool:

        return self.client is not None


    # =====================================================
    # GROQ LLM CALL
    # =====================================================

    def _llm(
        self,
        system: str,
        user: str,
        max_tokens: int = 800
    ) -> str:

        if not self.client:

            raise RuntimeError(
                "Groq client is not configured."
            )

        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model,

                messages=[
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": user
                    }
                ],

                max_tokens=max_tokens,

                temperature=0.2
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )


    # =====================================================
    # RAG QUESTION ANSWERING
    # =====================================================

    def ask(
        self,
        question: str,
        k: int = 3,
        today: date | None = None
    ) -> dict:

        today = (
            today
            or date.today()
        )

        hits = (
            self.retriever.search(
                question,
                k=k
            )
        )

        sources = [
            hit[0]
            for hit in hits
        ]

        # ---------------------------------------------
        # No matching information
        # ---------------------------------------------

        if not sources:

            return {
                "answer":
                    "I couldn't find that information "
                    "in the scholarship and exam "
                    "knowledge base. Please contact "
                    "the Scholarship Cell.",

                "sources": []
            }


        # ---------------------------------------------
        # Offline mode
        # ---------------------------------------------

        if not self.online:

            return {
                "answer":
                    "(Offline mode - matching information)\n\n"
                    + "\n\n".join(
                        sources
                    ),

                "sources":
                    sources
            }


        # ---------------------------------------------
        # Build context
        # ---------------------------------------------

        context = "\n\n".join(
            f"[{i + 1}] {source}"
            for i, source in enumerate(
                sources
            )
        )


        # ---------------------------------------------
        # System prompt
        # ---------------------------------------------

        system = (
            "You are a helpful scholarship and exam "
            "information assistant for a college. "

            "Answer ONLY using the provided context. "

            f"Today's date is {today.isoformat()}. "

            "If a deadline in the context has already "
            "passed, clearly mention that. "

            "If the answer is not present in the context, "
            "say that the information is unavailable "
            "and suggest contacting the Scholarship Cell. "

            "Do not invent scholarship names, amounts, "
            "dates, eligibility requirements or documents. "

            "Be concise and accurate."
        )


        # ---------------------------------------------
        # Groq
        # ---------------------------------------------

        answer = self._llm(
            system,
            f"""
Context:

{context}

Question:

{question}
""",
            max_tokens=800
        )


        return {
            "answer": answer,
            "sources": sources
        }


    # =====================================================
    # ELIGIBILITY CHECKER
    # =====================================================

    def check_eligibility(
        self,
        cgpa: float,
        family_income: float,
        category: str,
        today: date | None = None
    ) -> dict:

        today = (
            today
            or date.today()
        )

        eligible = []

        not_eligible = []


        for record in self.records:

            if record.kind != "scholarship":

                continue


            reasons = []


            # -----------------------------------------
            # CGPA
            # -----------------------------------------

            if cgpa < record.min_cgpa:

                reasons.append(
                    f"CGPA {cgpa} is below "
                    f"the required "
                    f"{record.min_cgpa}"
                )


            # -----------------------------------------
            # Income
            # -----------------------------------------

            if (
                record.max_income is not None
                and family_income > record.max_income
            ):

                reasons.append(
                    "Family income exceeds "
                    f"the limit of "
                    f"{int(record.max_income):,}"
                )


            # -----------------------------------------
            # Category
            # -----------------------------------------

            if (
                "all" not in record.categories
                and category.lower()
                not in record.categories
            ):

                reasons.append(
                    "Only for categories: "
                    + ", ".join(
                        sorted(
                            c.upper()
                            for c in record.categories
                        )
                    )
                )


            # -----------------------------------------
            # Not eligible
            # -----------------------------------------

            if reasons:

                not_eligible.append(
                    {
                        "name":
                            record.name,

                        "reasons":
                            reasons
                    }
                )

                continue


            # -----------------------------------------
            # Deadline status
            # -----------------------------------------

            open_ = (
                record.deadline is None
                or record.deadline >= today
            )


            eligible.append(
                {
                    "name":
                        record.name,

                    "amount":
                        record.fields.get(
                            "amount",
                            "-"
                        ),

                    "deadline":
                        record.deadline,

                    "status":
                        "Open"
                        if open_
                        else "Closed",

                    "documents":
                        record.fields.get(
                            "documents",
                            "-"
                        ),

                    "how_to_apply":
                        record.fields.get(
                            "how to apply",
                            "-"
                        )
                }
            )


        # Open scholarships first
        eligible.sort(
            key=lambda e: (
                e["status"] != "Open",
                e["deadline"]
                or date.max
            )
        )


        return {
            "eligible":
                eligible,

            "not_eligible":
                not_eligible
        }


    # =====================================================
    # UPCOMING EVENTS
    # =====================================================

    def upcoming_events(
        self,
        days: int = 60,
        today: date | None = None
    ) -> list[dict]:

        today = (
            today
            or date.today()
        )

        limit = (
            today
            + timedelta(
                days=days
            )
        )

        events = []


        for record in self.records:

            for label, event_date in [

                (
                    "Application/Registration deadline",
                    record.deadline
                ),

                (
                    "Exam date",
                    record.exam_date
                )

            ]:

                if (
                    event_date
                    and today <= event_date <= limit
                ):

                    events.append(
                        {
                            "date":
                                event_date,

                            "days_left":
                                (
                                    event_date
                                    - today
                                ).days,

                            "name":
                                record.name,

                            "type":
                                (
                                    record.kind.title()
                                    or "-"
                                ),

                            "event":
                                label
                        }
                    )


        return sorted(
            events,
            key=lambda e:
                e["date"]
        )