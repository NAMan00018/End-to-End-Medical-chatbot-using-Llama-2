from flask import Flask, render_template, jsonify, request
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import ctransformers
from langchain.chains import retrieval_qa

DB_FAISS_PATH="vectorstores/db_faiss"
app = Flask(__name__)


custom_prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    PROMPT=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

    return PROMPT

def load_llm():
    llm=ctransformers(
        model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config={'max_new_tokens':512,'temperature':0.8}
    )
    return llm


def retrieval_qa_chain(llm,PROMPT,db):
    qa_chain=retrieval_qa.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True, 
        chain_type_kwargs={"prompt": PROMPT})
    return qa_chain

def qa_bot():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device':'cpu'})
    db=FAISS.load_local(DB_FAISS_PATH,embeddings)
    llm=load_llm()
    qa_prompt=set_custom_prompt()
    qa=retrieval_qa_chain(llm,qa_prompt,db)

    return qa

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa_bot({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)