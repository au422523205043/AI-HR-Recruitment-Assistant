import os
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .utils import extract_pdf_text, candidate_name_from_filename
from .rag import build_vector_store, retrieve_candidate_evidence
from .schemas import CandidateScore, InterviewQuestions


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI HR Recruitment Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# OPENAI CONFIGURATION
# ============================================================

# Streamlit Cloud Secrets
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


OPENAI_MODEL = st.secrets.get(
    "OPENAI_MODEL",
    os.getenv("OPENAI_MODEL", "gpt-4o-mini")
)


# Check API key
if not OPENAI_API_KEY:
    st.error(
        "❌ OPENAI_API_KEY is missing. "
        "Please add it in Streamlit Cloud → Settings → Secrets."
    )
    st.stop()


# ============================================================
# LLM FUNCTION
# ============================================================

def llm():
    """
    Creates the OpenAI Chat Model.
    """

    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.1,
        api_key=OPENAI_API_KEY
    )


# ============================================================
# CANDIDATE SCREENING
# ============================================================

def screen_candidate(candidate, job_description, evidence):

    model = llm().with_structured_output(CandidateScore)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an evidence-based recruitment screening assistant.

Evaluate the candidate only according to job-related requirements.

Important rules:
- Do not infer protected characteristics.
- Do not use protected characteristics or their proxies.
- Do not invent experience, skills, education, or achievements.
- Base your evaluation only on the provided resume evidence.
- Compare the candidate's evidence with the job description.
- Give a fit score from 0 to 100.
- Identify strengths.
- Identify gaps.
- Identify matched requirements.
- Identify missing requirements.
- Provide a concise recommendation.
- This system is decision support for a human recruiter.
"""
        ),
        (
            "human",
            """
Candidate:
{candidate}

Job Description:
{job}

Retrieved Resume Evidence:
{evidence}
"""
        )
    ])

    messages = prompt.format_messages(
        candidate=candidate,
        job=job_description,
        evidence="\n---\n".join(evidence)
    )

    return model.invoke(messages)


# ============================================================
# INTERVIEW QUESTION GENERATOR
# ============================================================

def generate_interview_questions(
    candidate,
    job_description,
    evidence,
    score
):

    model = llm().with_structured_output(InterviewQuestions)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert technical recruitment assistant.

Generate exactly 8 practical and job-relevant interview questions.

Include a mixture of:
- Technical questions
- Behavioral questions
- Scenario-based questions

Questions should:
- Be based on the job description.
- Be based on the candidate evidence.
- Probe important claims.
- Explore identified gaps.
- Avoid protected or personal characteristics.
- Help a human recruiter evaluate the candidate fairly.
"""
        ),
        (
            "human",
            """
Candidate:
{candidate}

Job Description:
{job}

Screening Result:
{score}

Retrieved Resume Evidence:
{evidence}
"""
        )
    ])

    messages = prompt.format_messages(
        candidate=candidate,
        job=job_description,
        score=score.model_dump_json(),
        evidence="\n---\n".join(evidence)
    )

    return model.invoke(messages)


# ============================================================
# APPLICATION UI
# ============================================================

st.title("🤖 AI HR Recruitment Assistant")

st.caption(
    "Agent + Tools + RAG for resume screening, matching "
    "and interview preparation"
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

job = st.text_area(
    "📋 Job Description",
    height=240,
    placeholder="Enter the job description here..."
)


# ============================================================
# RESUME UPLOAD
# ============================================================

files = st.file_uploader(
    "📄 Upload Candidate Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = {}

if "evidence" not in st.session_state:
    st.session_state.evidence = {}

if "questions" not in st.session_state:
    st.session_state.questions = None


# ============================================================
# SCREEN CANDIDATES BUTTON
# ============================================================

if st.button(
    "🚀 Screen Candidates",
    type="primary",
    use_container_width=True
):

    # Validate input
    if not job.strip():
        st.error("❌ Please provide a job description.")
        st.stop()

    if not files:
        st.error("❌ Please upload at least one resume.")
        st.stop()


    # ========================================================
    # EXTRACT RESUMES
    # ========================================================

    resumes = {}

    with st.spinner("📄 Reading resumes..."):

        for file in files:

            try:

                text = extract_pdf_text(file.getvalue())

                if text and text.strip():

                    candidate_name = candidate_name_from_filename(
                        file.name
                    )

                    resumes[candidate_name] = text

            except Exception as e:

                st.warning(
                    f"Could not read {file.name}: {str(e)}"
                )


    # Check resumes
    if not resumes:

        st.error(
            "❌ No readable PDF text was found in the uploaded resumes."
        )

        st.stop()


    # ========================================================
    # BUILD RAG VECTOR STORE
    # ========================================================

    with st.spinner("🔍 Building RAG index..."):

        try:

            store = build_vector_store(resumes)

        except Exception as e:

            st.error(
                f"❌ Error while building RAG index: {str(e)}"
            )

            st.stop()


    # ========================================================
    # SCREEN EACH CANDIDATE
    # ========================================================

    results = {}
    evidence_map = {}

    progress = st.progress(0)

    total = len(resumes)

    for index, candidate in enumerate(resumes):

        with st.spinner(
            f"🤖 Screening {candidate}..."
        ):

            try:

                # Retrieve relevant resume evidence
                evidence = retrieve_candidate_evidence(
                    store,
                    job,
                    candidate
                )

                # AI candidate screening
                result = screen_candidate(
                    candidate,
                    job,
                    evidence
                )

                results[candidate] = result
                evidence_map[candidate] = evidence

            except Exception as e:

                st.error(
                    f"Error screening {candidate}: {str(e)}"
                )

        progress.progress(
            (index + 1) / total
        )


    # Save results
    st.session_state.results = results
    st.session_state.evidence = evidence_map

    st.success(
        "✅ Candidate screening completed successfully!"
    )


# ============================================================
# DISPLAY CANDIDATE RANKING
# ============================================================

if st.session_state.results:

    st.divider()

    st.subheader("🏆 Candidate Ranking")


    # Sort candidates by score
    ranked = sorted(
        st.session_state.results.values(),
        key=lambda x: x.overall_score,
        reverse=True
    )


    # ========================================================
    # DISPLAY EACH CANDIDATE
    # ========================================================

    for rank, result in enumerate(ranked, 1):

        with st.container(border=True):

            col1, col2, col3 = st.columns(
                [1, 5, 2]
            )


            # Rank
            with col1:

                st.metric(
                    "Rank",
                    rank
                )


            # Candidate details
            with col2:

                st.markdown(
                    f"### 👤 {result.candidate_name}"
                )

                st.write(
                    result.summary
                )


            # Score
            with col3:

                st.metric(
                    "Fit Score",
                    f"{result.overall_score:.0f}/100"
                )

                st.write(
                    f"**Recommendation:** "
                    f"{result.recommendation}"
                )


            # Strengths and gaps
            strength_col, gap_col = st.columns(2)


            with strength_col:

                st.markdown(
                    "#### ✅ Strengths"
                )

                for strength in result.strengths:

                    st.write(
                        "• " + strength
                    )


            with gap_col:

                st.markdown(
                    "#### ⚠️ Gaps"
                )

                for gap in result.gaps:

                    st.write(
                        "• " + gap
                    )


            # Matched requirements
            if hasattr(result, "matched_requirements"):

                with st.expander(
                    "✅ Matched Requirements"
                ):

                    for item in result.matched_requirements:

                        st.write(
                            "• " + item
                        )


            # Missing requirements
            if hasattr(result, "missing_requirements"):

                with st.expander(
                    "❌ Missing Requirements"
                ):

                    for item in result.missing_requirements:

                        st.write(
                            "• " + item
                        )


            # Retrieved evidence
            with st.expander(
                "🔎 Retrieved Resume Evidence"
            ):

                candidate_evidence = (
                    st.session_state.evidence.get(
                        result.candidate_name,
                        []
                    )
                )

                if candidate_evidence:

                    for evidence in candidate_evidence:

                        st.info(evidence)

                else:

                    st.write(
                        "No evidence retrieved."
                    )


    # ========================================================
    # INTERVIEW QUESTION GENERATOR
    # ========================================================

    st.divider()

    st.subheader(
        "🎤 Interview Question Generator"
    )


    selected_candidate = st.selectbox(
        "Select Candidate",
        [
            result.candidate_name
            for result in ranked
        ]
    )


    if st.button(
        "🧠 Generate Interview Questions",
        use_container_width=True
    ):

        with st.spinner(
            "Generating interview questions..."
        ):

            try:

                selected_evidence = (
                    st.session_state.evidence[
                        selected_candidate
                    ]
                )

                selected_score = (
                    st.session_state.results[
                        selected_candidate
                    ]
                )


                questions = generate_interview_questions(
                    selected_candidate,
                    job,
                    selected_evidence,
                    selected_score
                )


                st.session_state.questions = questions

                st.success(
                    "✅ Interview questions generated!"
                )

            except Exception as e:

                st.error(
                    f"❌ Error generating questions: {str(e)}"
                )


    # ========================================================
    # DISPLAY QUESTIONS
    # ========================================================

    if st.session_state.questions:

        questions = st.session_state.questions

        st.markdown(
            "### 📝 Interview Questions"
        )


        for index, question in enumerate(
            questions.questions,
            1
        ):

            st.markdown(
                f"**{index}. {question}**"
            )


        st.markdown(
            "### 🎯 Evaluation Focus"
        )


        for item in questions.evaluation_focus:

            st.write(
                "• " + item
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚖️ Human review is required for employment decisions."
)
