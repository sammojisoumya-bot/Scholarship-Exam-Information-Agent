"""
Streamlit UI for the Scholarship & Exam Information Agent.

Run:
    streamlit run app.py
"""

import os
from datetime import date

import pandas as pd
import streamlit as st

from rag_agent import CATEGORIES, ScholarshipAgent


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Scholarship & Exam Agent",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Scholarship & Exam Information Agent")
st.caption("RAG + TF-IDF + Groq AI")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Leave empty to use offline mode."
    )

    model = st.text_input(
        "Groq Model",
        value=os.getenv(
            "LLM_MODEL",
            "openai/gpt-oss-20b"
        )
    )


# =========================================================
# CREATE AGENT
# =========================================================

@st.cache_resource(show_spinner=False)
def get_agent(
    key: str,
    model_name: str
) -> ScholarshipAgent:

    return ScholarshipAgent(
        "college_info.txt",
        api_key=key or None,
        model=model_name
    )


agent = get_agent(
    api_key,
    model
)


# =========================================================
# AI STATUS
# =========================================================

if agent.online:

    st.sidebar.success(
        "🟢 Groq AI Mode: ON"
    )

else:

    st.sidebar.warning(
        "🟡 Offline Mode"
    )


# =========================================================
# TABS
# =========================================================

tab_chat, tab_elig, tab_dates = st.tabs(
    [
        "💬 Ask a Question",
        "✅ Eligibility Checker",
        "📅 Deadlines & Exams"
    ]
)


# =========================================================
# TAB 1 — CHAT / RAG
# =========================================================

with tab_chat:

    st.header("💬 Scholarship & Exam Assistant")

    st.write(
        "Ask questions about scholarships, eligibility, "
        "documents, exams, deadlines and application procedures."
    )

    # Store conversation
    st.session_state.setdefault(
        "messages",
        []
    )

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # Chat input
    question = st.chat_input(
        "e.g. What documents are needed for the merit scholarship?"
    )

    if question:

        # Display user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching scholarship information..."
            ):

                result = agent.ask(
                    question
                )

            st.markdown(
                result["answer"]
            )

            # Show retrieved sources
            if result["sources"]:

                with st.expander(
                    "📚 Retrieved Sources"
                ):

                    for source in result["sources"]:

                        st.info(source)

        # Save answer
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"]
            }
        )


# =========================================================
# TAB 2 — ELIGIBILITY CHECKER
# =========================================================

with tab_elig:

    st.header("✅ Scholarship Eligibility Checker")

    st.write(
        "Enter your academic and financial details "
        "to check scholarship eligibility."
    )

    with st.form(
        "eligibility"
    ):

        c1, c2, c3 = st.columns(3)

        cgpa = c1.number_input(
            "Current CGPA",
            0.0,
            10.0,
            7.5,
            step=0.1
        )

        income = c2.number_input(
            "Annual Family Income (₹)",
            0,
            10_000_000,
            400_000,
            step=10_000
        )

        category = c3.selectbox(
            "Category",
            CATEGORIES
        )

        go = st.form_submit_button(
            "🔍 Check Eligibility",
            type="primary"
        )


    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    if go:

        result = agent.check_eligibility(
            cgpa,
            income,
            category
        )

        eligible = result["eligible"]

        st.subheader(
            f"🎓 You are eligible for {len(eligible)} scholarship(s)"
        )

        if not eligible:

            st.warning(
                "No matching scholarships were found."
            )

        for scholarship in eligible:

            with st.container(
                border=True
            ):

                if scholarship["status"] == "Open":

                    badge = "🟢 Open"

                else:

                    badge = "🔴 Closed"

                st.markdown(
                    f"### {scholarship['name']} — {badge}"
                )

                st.write(
                    f"💰 Amount: {scholarship['amount']}"
                )

                st.write(
                    f"⏰ Deadline: "
                    f"{scholarship['deadline'] or 'Not specified'}"
                )

                st.caption(
                    f"📄 Documents: "
                    f"{scholarship['documents']}"
                )

                st.caption(
                    f"📝 How to apply: "
                    f"{scholarship['how_to_apply']}"
                )


        # -------------------------------------------------
        # NOT ELIGIBLE
        # -------------------------------------------------

        if result["not_eligible"]:

            with st.expander(
                "❌ Why are some scholarships not available?"
            ):

                for item in result["not_eligible"]:

                    st.markdown(
                        f"**{item['name']}**"
                    )

                    for reason in item["reasons"]:

                        st.write(
                            f"- {reason}"
                        )


# =========================================================
# TAB 3 — DEADLINES & EXAMS
# =========================================================

with tab_dates:

    st.header(
        "📅 Scholarship Deadlines & Exam Dates"
    )

    days = st.slider(
        "Show events in the next N days",
        7,
        180,
        60
    )

    events = agent.upcoming_events(
        days=days
    )

    st.caption(
        f"Today: {date.today().isoformat()}"
    )

    if events:

        df = pd.DataFrame(
            events
        ).rename(
            columns={
                "date": "Date",
                "days_left": "Days Left",
                "name": "Name",
                "type": "Type",
                "event": "Event"
            }
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No deadlines or exams in this window."
        )