from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store(resumes):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    docs=[]
    for candidate,text in resumes.items():
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"candidate":candidate}))
    return FAISS.from_documents(docs, OpenAIEmbeddings())

def retrieve_candidate_evidence(store, job_description, candidate, k=8):
    hits=store.similarity_search(job_description,k=max(12,k*3))
    return [d.page_content for d in hits if d.metadata.get("candidate")==candidate][:k]
