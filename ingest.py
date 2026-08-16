import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import docx

DOCUMENTS_DIR="documents"
DB_DIR="chroma_db"
CHUNK_SIZE=1000
CHUNK_OVERLAP=200


def read_txt(path):
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        return f.read()


def read_pdf(path):
    reader=PdfReader(path)
    text=""
    for page in reader.pages:
        text += page.extract_text() or ""
        text += "\n"
    return text


def read_docx(path):
    doc=docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_document(path):
    ext=os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return read_txt(path)
    elif ext == ".pdf":
        return read_pdf(path)
    elif ext == ".docx":
        return read_docx(path)
    else:
        print(f"  Atlanıyor (desteklenmeyen format): {path}")
        return None


def chunk_text(text,chunk_size=CHUNK_SIZE,overlap=CHUNK_OVERLAP):
    chunks=[]
    start=0
    text_length=len(text)
    while start < text_length:
        end=start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def main():
    print("Embedding fonksiyonu yükleniyor (ilk çalıştırmada model indirilir)...")
    embedding_fn=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client=chromadb.PersistentClient(path=DB_DIR)
    collection=client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_fn
    )

    if not os.path.isdir(DOCUMENTS_DIR):
        print(f"HATA: '{DOCUMENTS_DIR}' klasörü bulunamadı. Önce oluşturup içine dosya koy.")
        return

    files=[f for f in os.listdir(DOCUMENTS_DIR) if not f.startswith(".")]
    if not files:
        print(f"'{DOCUMENTS_DIR}' klasörü boş. İçine PDF/txt/docx dosyası koy.")
        return

    total_chunks=0
    for filename in files:
        path=os.path.join(DOCUMENTS_DIR,filename)
        print(f"\nOkunuyor: {filename}")
        text=load_document(path)
        if not text:
            continue

        chunks=chunk_text(text)
        if not chunks:
            print("  İçerik boş, atlanıyor.")
            continue

        ids=[f"{filename}_{i}" for i in range(len(chunks))]
        metadatas=[{"source": filename,"chunk_index": i} for i in range(len(chunks))]

        collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        print(f"  {len(chunks)} parça eklendi.")
        total_chunks += len(chunks)

    print(f"\nTamamlandı. Toplam {total_chunks} parça '{DB_DIR}' klasöründeki veritabanına kaydedildi.")


if __name__ == "__main__":
    main()