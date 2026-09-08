import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .schemas import CandidateScore, InterviewQuestions

def llm():
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),temperature=0.1)

def screen_candidate(candidate, job_description, evidence):
    model=llm().with_structured_output(CandidateScore)
    prompt=ChatPromptTemplate.from_messages([
        ("system", """You are an evidence-based recruitment screening assistant.
Evaluate only job-related requirements. Do not infer or use protected characteristics or proxies.
Do not invent experience. Base claims on retrieved resume evidence.
Return a 0-100 fit score, strengths, gaps, matched/missing requirements and a concise recommendation.
This is decision support for a human recruiter."""),
        ("human","""Candidate: {candidate}
Job Description:
{job}
Retrieved Resume Evidence:
{evidence}""")])
    return model.invoke(prompt.format_messages(candidate=candidate,job=job_description,evidence="\n---\n".join(evidence)))

def generate_interview_questions(candidate, job_description, evidence, score):
    model=llm().with_structured_output(InterviewQuestions)
    prompt=ChatPromptTemplate.from_messages([
        ("system","""Generate 8 practical, job-relevant interview questions based on the job description and candidate evidence.
Mix technical, behavioral and scenario questions. Probe claims and gaps. Avoid protected or personal characteristics."""),
        ("human","""Candidate: {candidate}
Job: {job}
Screening result: {score}
Evidence: {evidence}""")])
    return model.invoke(prompt.format_messages(candidate=candidate,job=job_description,score=score.model_dump_json(),evidence="\n---\n".join(evidence)))
