import os
from flask import Flask, render_template, request, redirect, url_for
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from langchain.schema import Document


app = Flask(__name__)

load_dotenv('.env')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize global variables for Qdrant and embeddings
qdrant = None
embedding_model = None

# Initialize the embedding model and Qdrant
def initialize_services():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_size = len(embedding_model.embed_query("test"))
    qdrant = QdrantClient(":memory:")
    qdrant.recreate_collection(
        collection_name="documents",
        vectors_config={"size": vector_size, "distance": "Cosine"}
    )
    return qdrant, embedding_model

# Function to convert PDF and add documents to Qdrant
def add_documents_to_qdrant():
    source = "../pdf/จังหวัดน่าน.pdf"  # Path to the PDF file
    converter = DocumentConverter()
    result = converter.convert(source)

    markdown_text = result.document.export_to_markdown()
    doc = Document(page_content=markdown_text)  # Ensure Document is correctly imported
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = splitter.split_documents([doc])

    texts = [doc.page_content for doc in document_chunks]
    vectors = embedding_model.embed_documents(texts)
    points = [PointStruct(id=i, vector=vectors[i], payload={"text": texts[i]}) for i in range(len(texts))]
    qdrant.upsert(collection_name="documents", points=points)

# Document search function
def search_documents(query, qdrant, embedding_model):
    query_vector = embedding_model.embed_query(query)
    search_results = qdrant.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=4  # Adjust as necessary
    )
    if not search_results:
        return []
    return [hit.payload.get("text", "เอกสารไม่มีข้อความ") for hit in search_results]

# Generate an answer using Groq
def generate_answer(query, qdrant, embedding_model):
    retrieved_docs = search_documents(query, qdrant, embedding_model)
    if not retrieved_docs:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"  # "No relevant information found"
    
    context = "\n".join([str(doc) for doc in retrieved_docs if isinstance(doc, str)])
    if not context.strip():
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"
    
    prompt = f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {query}\n\nคำตอบ:"
    groq_client = Groq(api_key=GROQ_API_KEY)
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการสร้างคำตอบ: {str(e)}"  # "Error occurred in generating the answer"

# Route for the home page
@app.route("/", methods=["GET", "POST"])
def index():
    global qdrant, embedding_model
    if not qdrant or not embedding_model:  # Initialize services only once
        qdrant, embedding_model = initialize_services()
        add_documents_to_qdrant()

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            answer = generate_answer(query, qdrant, embedding_model)
            return render_template("index.html", query=query, answer=answer)
    
    return render_template("index.html", query=None, answer=None)

if __name__ == "__main__":
    app.run(debug=True)