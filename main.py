import os
import io
import re
import json
import streamlit as st

from pypdf import PdfReader

from openai import OpenAI

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AI HR Recruitment Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# OPENAI CONFIGURATION
# ============================================================

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    OPENAI_MODEL = st.secrets.get(
        "OPENAI_MODEL",
        "gpt-4o-mini"
    )
except Exception:
    OPENAI_MODEL = os.getenv(
        "OPENAI_MODEL",
        "gpt-4o-mini"
    )


if not OPENAI_API_KEY:

    st.error(
        "❌ OPENAI_API_KEY is missing."
    )

    st.info(
        "Go to Streamlit Cloud → Settings → Secrets "
        "and add your OpenAI API key."
    )

    st.stop()


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🤖 AI HR Recruitment Assistant")

st.caption(
    "AI-powered resume screening, candidate matching, "
    "RAG-based evidence retrieval and interview preparation"
)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_bytes):

    try:

        pdf_file = io.BytesIO(pdf_bytes)

        reader = PdfReader(pdf_file)

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    except Exception as e:

        return ""


# ============================================================
# CANDIDATE NAME
# ============================================================

def candidate_name_from_filename(filename):

    name = os.path.splitext(filename)[0]

    name = re.sub(
        r"[_\-]+",
        " ",
        name
    )

    name = re.sub(
        r"\s+",
        " ",
        name
    )

    return name.strip().title()


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=1200,
    overlap=200
):

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(
                chunk.strip()
            )

        start = end - overlap

        if start < 0:
            start = 0

        if end >= len(text):
            break

    return chunks


# ============================================================
# BUILD RAG VECTOR STORE
# ============================================================

def build_vector_store(resumes):

    documents = []

    for candidate, text in resumes.items():

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):

            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "candidate": candidate,
                        "chunk": index
                    }
                )
            )

    if not documents:

        return None

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    return vector_store


# ============================================================
# RETRIEVE CANDIDATE EVIDENCE
# ============================================================

def retrieve_candidate_evidence(
    vector_store,
    job_description,
    candidate,
    k=6
):

    if vector_store is None:
        return []

    query = f"""
Find resume evidence for candidate {candidate}
that is relevant to the following job description.

Job Description:
{job_description}

Focus on:
skills, technologies, education, projects,
work experience, internships, certifications,
achievements and job-related qualifications.
"""

    try:

        results = vector_store.similarity_search(
            query,
            k=k
        )

        candidate_results = [
            doc.page_content
            for doc in results
            if doc.metadata.get("candidate") == candidate
        ]

        if candidate_results:
            return candidate_results

        return [
            doc.page_content
            for doc in results
        ]

    except Exception:

        return []


# ============================================================
# OPENAI JSON HELPER
# ============================================================

def call_openai_json(
    system_prompt,
    user_prompt
):

    try:

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.1,
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        content = (
            response.choices[0]
            .message
            .content
        )

        return json.loads(content)

    except Exception as e:

        raise RuntimeError(
            f"OpenAI API error: {str(e)}"
        )


# ============================================================
# SCREEN CANDIDATE
# ============================================================

def screen_candidate(
    candidate,
    job_description,
    evidence
):

    evidence_text = "\n---\n".join(
        evidence
    )

    system_prompt = """
You are an evidence-based recruitment screening assistant.

Evaluate candidates ONLY using job-related information.

Rules:

1. Do not infer protected characteristics.
2. Do not use protected characteristics or proxies.
3. Do not invent experience or skills.
4. Use only the provided resume evidence.
5. Compare the candidate against the job description.
6. Give a fit score from 0 to 100.
7. Identify strengths.
8. Identify gaps.
9. Identify matched requirements.
10. Identify missing requirements.
11. Give a concise recommendation.
12. This is decision support for a human recruiter.

Return ONLY valid JSON using this structure:

{
    "candidate_name": "string",
    "overall_score": 0,
    "summary": "string",
    "strengths": ["string"],
    "gaps": ["string"],
    "matched_requirements": ["string"],
    "missing_requirements": ["string"],
    "recommendation": "string"
}

The overall_score must be a number between 0 and 100.
"""

    user_prompt = f"""
Candidate:
{candidate}

Job Description:
{job_description}

Retrieved Resume Evidence:
{evidence_text}
"""

    result = call_openai_json(
        system_prompt,
        user_prompt
    )

    result["candidate_name"] = candidate

    try:

        result["overall_score"] = float(
            result.get(
                "overall_score",
                0
            )
        )

    except Exception:

        result["overall_score"] = 0

    return result


# ============================================================
# INTERVIEW QUESTION GENERATOR
# ============================================================

def generate_interview_questions(
    candidate,
    job_description,
    evidence,
    score
):

    evidence_text = "\n---\n".join(
        evidence
    )

    system_prompt = """
You are an expert recruitment interview assistant.

Generate exactly 8 practical and job-relevant
interview questions.

Include a mixture of:

- Technical questions
- Behavioral questions
- Scenario-based questions

Questions must:

- Be based on the job description.
- Be based on candidate evidence.
- Probe important candidate claims.
- Explore relevant gaps.
- Avoid protected or personal characteristics.
- Help a human recruiter evaluate the candidate fairly.

Return ONLY valid JSON:

{
    "questions": [
        "question 1",
        "question 2",
        "question 3",
        "question 4",
        "question 5",
        "question 6",
        "question 7",
        "question 8"
    ],
    "evaluation_focus": [
        "focus 1",
        "focus 2",
        "focus 3",
        "focus 4"
    ]
}
"""

    user_prompt = f"""
Candidate:
{candidate}

Job Description:
{job_description}

Screening Result:
{json.dumps(score, indent=2)}

Retrieved Resume Evidence:
{evidence_text}
"""

    result = call_openai_json(
        system_prompt,
        user_prompt
    )

    return result


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.subheader("📋 Job Description")

job = st.text_area(
    "Enter the job description",
    height=240,
    placeholder=(
        "Example:\n"
        "We are looking for a Python developer "
        "with experience in Django, REST APIs, "
        "SQL and cloud technologies..."
    )
)


# ============================================================
# RESUME UPLOAD
# ============================================================

st.subheader("📄 Candidate Resumes")

files = st.file_uploader(
    "Upload candidate resumes",
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
# SCREEN CANDIDATES
# ============================================================

if st.button(
    "🚀 Screen Candidates",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not job.strip():

        st.error(
            "❌ Please enter a job description."
        )

        st.stop()


    if not files:

        st.error(
            "❌ Please upload at least one resume."
        )

        st.stop()


    # --------------------------------------------------------
    # EXTRACT RESUMES
    # --------------------------------------------------------

    resumes = {}

    progress = st.progress(
        0,
        text="Reading resumes..."
    )

    total_files = len(files)

    for index, file in enumerate(files):

        text = extract_pdf_text(
            file.getvalue()
        )

        if text.strip():

            candidate = (
                candidate_name_from_filename(
                    file.name
                )
            )

            resumes[candidate] = text

        progress.progress(
            (index + 1) / total_files,
            text=f"Reading {file.name}"
        )


    progress.empty()


    if not resumes:

        st.error(
            "❌ No readable text was found "
            "in the uploaded PDF files."
        )

        st.stop()


    st.success(
        f"✅ {len(resumes)} resume(s) successfully read."
    )


    # --------------------------------------------------------
    # BUILD RAG
    # --------------------------------------------------------

    with st.spinner(
        "🔍 Building RAG resume index..."
    ):

        try:

            vector_store = build_vector_store(
                resumes
            )

        except Exception as e:

            st.error(
                f"❌ RAG error: {str(e)}"
            )

            st.stop()


    # --------------------------------------------------------
    # SCREEN CANDIDATES
    # --------------------------------------------------------

    results = {}

    evidence_map = {}

    progress = st.progress(
        0,
        text="Screening candidates..."
    )

    candidates = list(
        resumes.keys()
    )

    total_candidates = len(
        candidates
    )


    for index, candidate in enumerate(
        candidates
    ):

        try:

            evidence = (
                retrieve_candidate_evidence(
                    vector_store,
                    job,
                    candidate
                )
            )

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
            (index + 1) / total_candidates,
            text=f"Screening {candidate}"
        )


    progress.empty()


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    st.session_state.results = results

    st.session_state.evidence = evidence_map

    st.session_state.questions = None


    st.success(
        "🎉 Candidate screening completed!"
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.results:

    st.divider()

    st.subheader(
        "🏆 Candidate Ranking"
    )


    # --------------------------------------------------------
    # SORT BY SCORE
    # --------------------------------------------------------

    ranked = sorted(
        st.session_state.results.values(),
        key=lambda x: x.get(
            "overall_score",
            0
        ),
        reverse=True
    )


    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Candidates",
            len(ranked)
        )

    with col2:

        best_score = (
            ranked[0].get(
                "overall_score",
                0
            )
            if ranked
            else 0
        )

        st.metric(
            "Highest Fit",
            f"{best_score:.0f}/100"
        )

    with col3:

        if best_score >= 80:

            status = "Strong Match"

        elif best_score >= 60:

            status = "Moderate Match"

        else:

            status = "Low Match"

        st.metric(
            "Top Candidate",
            status
        )


    # --------------------------------------------------------
    # CANDIDATE CARDS
    # --------------------------------------------------------

    for rank, result in enumerate(
        ranked,
        1
    ):

        candidate_name = result.get(
            "candidate_name",
            "Unknown Candidate"
        )

        score = float(
            result.get(
                "overall_score",
                0
            )
        )


        with st.container(
            border=True
        ):

            col1, col2, col3 = st.columns(
                [1, 5, 2]
            )


            # Rank

            with col1:

                st.metric(
                    "Rank",
                    rank
                )


            # Candidate

            with col2:

                st.markdown(
                    f"### 👤 {candidate_name}"
                )

                st.write(
                    result.get(
                        "summary",
                        ""
                    )
                )


            # Score

            with col3:

                st.metric(
                    "Fit Score",
                    f"{score:.0f}/100"
                )

                st.write(
                    "**Recommendation:** "
                    + result.get(
                        "recommendation",
                        "N/A"
                    )
                )


            # ------------------------------------------------
            # STRENGTHS AND GAPS
            # ------------------------------------------------

            strength_col, gap_col = st.columns(2)


            with strength_col:

                st.markdown(
                    "#### ✅ Strengths"
                )

                strengths = result.get(
                    "strengths",
                    []
                )

                if strengths:

                    for item in strengths:

                        st.write(
                            "• " + str(item)
                        )

                else:

                    st.write(
                        "No strengths identified."
                    )


            with gap_col:

                st.markdown(
                    "#### ⚠️ Gaps"
                )

                gaps = result.get(
                    "gaps",
                    []
                )

                if gaps:

                    for item in gaps:

                        st.write(
                            "• " + str(item)
                        )

                else:

                    st.write(
                        "No major gaps identified."
                    )


            # ------------------------------------------------
            # MATCHED REQUIREMENTS
            # ------------------------------------------------

            with st.expander(
                "✅ Matched Requirements"
            ):

                matched = result.get(
                    "matched_requirements",
                    []
                )

                if matched:

                    for item in matched:

                        st.write(
                            "• " + str(item)
                        )

                else:

                    st.write(
                        "No matched requirements listed."
                    )


            # ------------------------------------------------
            # MISSING REQUIREMENTS
            # ------------------------------------------------

            with st.expander(
                "❌ Missing Requirements"
            ):

                missing = result.get(
                    "missing_requirements",
                    []
                )

                if missing:

                    for item in missing:

                        st.write(
                            "• " + str(item)
                        )

                else:

                    st.write(
                        "No missing requirements listed."
                    )


            # ------------------------------------------------
            # RETRIEVED RESUME EVIDENCE
            # ------------------------------------------------

            with st.expander(
                "🔎 Retrieved Resume Evidence"
            ):

                candidate_evidence = (
                    st.session_state.evidence.get(
                        candidate_name,
                        []
                    )
                )

                if candidate_evidence:

                    for evidence in candidate_evidence:

                        st.info(
                            evidence
                        )

                else:

                    st.write(
                        "No evidence retrieved."
                    )


# ============================================================
# INTERVIEW QUESTION GENERATOR
# ============================================================

if st.session_state.results:

    st.divider()

    st.subheader(
        "🎤 Interview Question Generator"
    )


    ranked_candidates = sorted(
        st.session_state.results.values(),
        key=lambda x: x.get(
            "overall_score",
            0
        ),
        reverse=True
    )


    candidate_names = [
        result.get(
            "candidate_name"
        )
        for result in ranked_candidates
    ]


    selected_candidate = st.selectbox(
        "Select Candidate",
        candidate_names
    )


    if st.button(
        "🧠 Generate Interview Questions",
        use_container_width=True
    ):

        selected_evidence = (
            st.session_state.evidence.get(
                selected_candidate,
                []
            )
        )

        selected_score = (
            st.session_state.results.get(
                selected_candidate,
                {}
            )
        )


        with st.spinner(
            "🤖 Generating interview questions..."
        ):

            try:

                questions = (
                    generate_interview_questions(
                        selected_candidate,
                        job,
                        selected_evidence,
                        selected_score
                    )
                )

                st.session_state.questions = questions

                st.success(
                    "✅ Interview questions generated!"
                )

            except Exception as e:

                st.error(
                    f"❌ Interview generation failed: {str(e)}"
                )


    # ========================================================
    # DISPLAY QUESTIONS
    # ========================================================

    if st.session_state.questions:

        questions = (
            st.session_state.questions
        )


        st.markdown(
            "### 📝 Interview Questions"
        )


        question_list = questions.get(
            "questions",
            []
        )


        for index, question in enumerate(
            question_list,
            1
        ):

            st.markdown(
                f"**{index}. {question}**"
            )


        st.markdown(
            "### 🎯 Evaluation Focus"
        )


        focus_list = questions.get(
            "evaluation_focus",
            []
        )


        for item in focus_list:

            st.write(
                "• " + str(item)
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚖️ Human review is required for employment decisions. "
    "The system provides decision support and should not "
    "make automated employment decisions."
)
