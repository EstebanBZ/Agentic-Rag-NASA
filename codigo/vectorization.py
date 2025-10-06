import os
import time
from tqdm import tqdm  # ✅ barra de progreso
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ nuevo import
from langchain_community.vectorstores import FAISS

# --------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# --------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "extracted_md")
OUTPUT_DIR = os.path.join(BASE_DIR, "vector_db")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------
# TEMPORIZADOR GLOBAL
# --------------------------------------------------------
global_start = time.time()

# --------------------------------------------------------
# CARGA DE DOCUMENTOS
# --------------------------------------------------------

start = time.time()
documents = []
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"❌ No se encontró el directorio {DATA_DIR}")

for file in os.listdir(DATA_DIR):
    if file.endswith(".md"):
        path = os.path.join(DATA_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            documents.append({"filename": file, "content": f.read()})

print(f"📄 {len(documents)} documentos Markdown cargados desde {DATA_DIR}")
print(f"⏱️ Tiempo de carga: {time.time() - start:.2f} s\n")

# --------------------------------------------------------
# CHUNKING (división del texto)
# --------------------------------------------------------

start = time.time()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=250,
    length_function=len
)

chunks = []
print("🧩 Dividiendo documentos en chunks...")
for doc in tqdm(documents, desc="Procesando documentos", unit="doc"):
    parts = splitter.split_text(doc["content"])
    for i, part in enumerate(parts):
        chunks.append({
            "filename": doc["filename"],
            "chunk_id": i,
            "content": part
        })

print(f"🧩 {len(chunks)} chunks generados en total")
print(f"⏱️ Tiempo de chunking: {time.time() - start:.2f} s\n")

# --------------------------------------------------------
# EMBEDDINGS (modelo Hugging Face)
# --------------------------------------------------------

start = time.time()
print("🧠 Cargando modelo de embeddings...")
model_name = "BAAI/bge-small-en-v1.5"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},  # cambia a 'cuda' si tienes GPU
    encode_kwargs={"normalize_embeddings": True}
)
print(f"✅ Modelo cargado en {time.time() - start:.2f} s\n")

# --------------------------------------------------------
# VECTOR STORE (FAISS)
# --------------------------------------------------------

start = time.time()
print("⚙️ Generando vectores y creando base FAISS...")

texts = [c["content"] for c in chunks]
metadatas = [{"source": c["filename"], "chunk_id": c["chunk_id"]} for c in chunks]

batch_size = 128
vectors = []
metas = []

for i in tqdm(range(0, len(texts), batch_size), desc="Calculando embeddings", unit="batch"):
    batch_texts = texts[i:i+batch_size]
    batch_metas = metadatas[i:i+batch_size]
    vectors.extend(batch_texts)
    metas.extend(batch_metas)

print("📊 Construyendo índice FAISS (esto puede tardar varios minutos)...")
build_start = time.time()
vectorstore = FAISS.from_texts(
    texts=vectors,
    embedding=embeddings,
    metadatas=metas
)
print(f"✅ Índice FAISS construido en {time.time() - build_start:.2f} s")

save_start = time.time()
vectorstore.save_local(OUTPUT_DIR)
print(f"💾 Índice guardado en {OUTPUT_DIR} ({time.time() - save_start:.2f} s)\n")
print(f"⏱️ Tiempo total de generación y guardado FAISS: {time.time() - start:.2f} s\n")

# --------------------------------------------------------
# PRUEBA DE BÚSQUEDA
# --------------------------------------------------------

start = time.time()
query = "How does microgravity affect plant development?"
results = vectorstore.similarity_search(query, k=3)
print(f"⏱️ Tiempo de búsqueda de prueba: {time.time() - start:.2f} s")

print("\n🔍 Resultados de búsqueda de prueba:")
for r in results:
    print(f"\n📘 Documento: {r.metadata['source']} (chunk {r.metadata['chunk_id']})")
    print(r.page_content[:300], "...")

# --------------------------------------------------------
# TIEMPO TOTAL GLOBAL
# --------------------------------------------------------

print(f"\n🚀 Tiempo total de ejecución: {time.time() - global_start:.2f} s")
