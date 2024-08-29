from flask import Flask, render_template, request
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain, StuffDocumentsChain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import PromptTemplate
from typing import Any, List, Optional
from ctransformers import AutoModelForCausalLM

DB_FAISS_PATH = "vectorstores/db_faiss"
app = Flask(__name__)

# Define the custom prompt template
custom_prompt_template = """
Use the following pieces of information to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

class CustomLLM(LLM):
    model: Any
    
    def __init__(self, model_path: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="llama"
        )
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        return self.model(prompt, stop=stop)

    @property
    def _llm_type(self) -> str:
        return "custom"

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm():
    return CustomLLM("model/llama-2-7b-chat.ggmlv3.q4_0.bin")

def retrieval_qa_chain(llm, prompt, db):
    document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )
    document_variable_name = "context"
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    )
    retriever = db.as_retriever(search_kwargs={'k': 2})
    return RetrievalQA(combine_documents_chain=stuff_chain, retriever=retriever)

def qa_bot(input_query):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    prompt_template = set_custom_prompt()
    qa = retrieval_qa_chain(llm, prompt_template, db)
    result = qa({"query": input_query})
    return result["result"]

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_query = msg
    print(input_query)
    response = qa_bot(input_query)
    print("Response : ", response)
    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)