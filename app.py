from flask import Flask, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

app = Flask(__name__)

# Load PDFs
loader1 = PyPDFLoader("act.pdf")
loader2 = PyPDFLoader("rules.pdf")
docs = loader1.load() + loader2.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query")

    llm = ChatOpenAI(temperature=0)

    prompt = f"""
আপনি একজন West Bengal Co-operative আইন বিশেষজ্ঞ।
বাংলায় উত্তর দিন, section ও practical example সহ।

প্রশ্ন: {user_query}
"""

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever()
    )

    result = qa.run(prompt)

    return jsonify({"answer": result})

@app.route("/")
def home():
    return "WB Co-op AI Backend Running"
