import os
import streamlit as st
from dotenv import load_dotenv
from .utils import extract_pdf_text,candidate_name_from_filename
from .rag import build_vector_store,retrieve_candidate_evidence
from .agents import screen_candidate,generate_interview_questions

load_dotenv()
st.set_page_config(page_title="AI HR Recruitment Assistant",page_icon="🤖",layout="wide")
st.title("🤖 AI HR Recruitment Assistant")
st.caption("Agent + Tools + RAG for resume screening, matching and interview preparation")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("Add OPENAI_API_KEY to .env before screening.")

job=st.text_area("Job Description",height=240)
files=st.file_uploader("Upload candidate resumes (PDF)",type=["pdf"],accept_multiple_files=True)

if "results" not in st.session_state: st.session_state.results={}
if "evidence" not in st.session_state: st.session_state.evidence={}

if st.button("🚀 Screen Candidates",type="primary",use_container_width=True):
    if not job.strip() or not files:
        st.error("Provide a job description and at least one resume."); st.stop()
    resumes={}
    for f in files:
        text=extract_pdf_text(f.getvalue())
        if text: resumes[candidate_name_from_filename(f.name)]=text
    if not resumes:
        st.error("No readable PDF text found."); st.stop()
    with st.spinner("Building RAG index..."): store=build_vector_store(resumes)
    results={}; evidence_map={}
    for candidate in resumes:
        with st.spinner(f"Screening {candidate}..."):
            ev=retrieve_candidate_evidence(store,job,candidate)
            results[candidate]=screen_candidate(candidate,job,ev)
            evidence_map[candidate]=ev
    st.session_state.results=results; st.session_state.evidence=evidence_map
    st.success("Screening complete.")

if st.session_state.results:
    st.subheader("Candidate Ranking")
    ranked=sorted(st.session_state.results.values(),key=lambda x:x.overall_score,reverse=True)
    for i,r in enumerate(ranked,1):
        with st.container(border=True):
            a,b,c=st.columns([1,5,2])
            a.metric("Rank",i); b.markdown(f"### {r.candidate_name}"); b.write(r.summary)
            c.metric("Fit",f"{r.overall_score:.0f}/100"); c.write(f"**{r.recommendation}**")
            x,y=st.columns(2)
            with x:
                st.markdown("**Strengths**")
                for s in r.strengths: st.write("• "+s)
            with y:
                st.markdown("**Gaps**")
                for g in r.gaps: st.write("• "+g)
            with st.expander("Retrieved evidence"):
                for e in st.session_state.evidence[r.candidate_name]: st.info(e)

    st.subheader("Interview Question Generator")
    selected=st.selectbox("Candidate",[r.candidate_name for r in ranked])
    if st.button("Generate Interview Questions"):
        with st.spinner("Generating questions..."):
            st.session_state.questions=generate_interview_questions(
                selected,job,st.session_state.evidence[selected],st.session_state.results[selected])
    if "questions" in st.session_state:
        q=st.session_state.questions
        for i,question in enumerate(q.questions,1): st.markdown(f"**{i}. {question}**")
        st.markdown("**Evaluation focus**")
        for item in q.evaluation_focus: st.write("• "+item)

st.divider()
st.caption("Human review is required for employment decisions.")
