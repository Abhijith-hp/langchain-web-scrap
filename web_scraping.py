import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import time
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

import os

OPENAPI_API_KEY = "OPENAI_API_KEY"
OPENAPI_ORG = "ORG_AI"
OPENAPI_PROJECT = "PROJECT_ID"

# Set OpenAI credentials in environment (for LangChain usage)
os.environ["OPENAI_API_KEY"] = OPENAPI_API_KEY
os.environ["OPENAI_ORGANIZATION"] = OPENAPI_ORG
os.environ["OPENAI_PROJECT"] = OPENAPI_PROJECT

VECTORSTORE_PATH = "geojit_faiss_index"

def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in {'http', 'https'}

def crawl_website(base_url, max_pages=50):
    visited = set()
    to_visit = [base_url]
    documents = []
    base_domain = urlparse(base_url).netloc

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue

        try:
            print(f"Crawling: {url}")
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            documents.append(Document(page_content=text, metadata={"source": url}))
            visited.add(url)

            for link in soup.find_all("a", href=True):
                next_url = urljoin(base_url, link["href"])
                next_url = urldefrag(next_url).url
                parsed_next = urlparse(next_url)

                if (
                    parsed_next.scheme in {"http", "https"}
                    and parsed_next.netloc == base_domain
                    and next_url not in visited
                    and next_url not in to_visit
                ):
                    to_visit.append(next_url)

            time.sleep(0.5)

        except requests.RequestException as e:
            print(f"Failed to crawl {url}: {e}")

    return documents

    


async def run_rag():
    import pdb;pdb.set_trace()
   
    if os.path.exists(VECTORSTORE_PATH):
        print("🔁 Loading existing FAISS index from disk...")
        vectorestore = FAISS.load_local(VECTORSTORE_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        print("🌐 Crawling and building FAISS index...")
        docs = crawl_website("weburl")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vectorestore = FAISS.from_documents(split_docs, embeddings)

      
        vectorestore.save_local(VECTORSTORE_PATH)
        print("💾 FAISS index saved locally.")

    retriever = vectorestore.as_retriever()
    llm = ChatOpenAI(temperature=0.7)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    question = "How to apply for IPO using Geojit?"
    result = qa_chain.invoke({"query": question})

    print(" Retrieval Augmented Generation (RAG) Result:")
    print(" Question:", question)
    print(" Answer:", result["result"])
    print(" Sources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata["source"])


# Run it using asyncio
if __name__ == "__main__":
    asyncio.run(run_rag())
