import chromadb
from chromadb.utils import embedding_functions
from foundry_local_sdk import Configuration,FoundryLocalManager
import re

DB_DIR="chroma_db"
MODEL_ALIAS="qwen3-8b"
TOP_K=5


def setup_retriever():
    embedding_fn=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client=chromadb.PersistentClient(path=DB_DIR)
    collection=client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_fn
    )
    return collection


def setup_llm():
    FoundryLocalManager.initialize(Configuration(app_name="local-rag-assistant"))
    manager=FoundryLocalManager.instance
    model=manager.catalog.get_model(MODEL_ALIAS)
    model.download()
    model.load()
    chat_client=model.get_chat_client()

    chat_client.settings.max_tokens=500
    chat_client.settings.temperature=0.1
    chat_client.settings.top_p=0.1

    return chat_client

def retrieve_context(collection,question,top_k=TOP_K):
    results=collection.query(
        query_texts=[question],
        n_results=top_k
    )
    documents=results["documents"][0]
    sources=[meta["source"] for meta in results["metadatas"][0]]
    return documents,sources


def build_prompt(question,context_chunks):
    context_text="\n\n---\n\n".join(context_chunks)
    prompt=f"""Aşağıdaki bağlamı kullanarak soruyu cevapla.

ÖRNEK (nasıl cevap vermen gerektiğini gösterir):
Bağlamda "Grup A'da oran %40, Grup B'de %20 idi (p=0.32)" yazıyorsa, DOĞRU cevap şudur:
"Grup A'da oran sayısal olarak daha yüksek olsa da (%40'a karşı %20), bu fark istatistiksel olarak anlamlı değildir (p=0.32, p>0.05 olduğu için)."
YANLIŞ cevap şudur: "Grup A'da anlamlı derecede yüksek bulunmuştur." (çünkü p=0.32, 0.05'ten büyük olduğu için anlamlı değildir)

KURAL: p < 0.05 ise "anlamlı", p >= 0.05 ise "anlamlı değil" de. Sayısal olarak yüksek görünse bile p>=0.05 ise anlamlı değildir.

Tıbbi terimleri ve kısaltmaları (CVD, OSAS, HT, DM gibi) çevirme veya karıştırma.

Eğer cevap bağlamda yoksa, "Bu bilgi dokümanlarda bulunmuyor" de. Kısa ve net ol.

Bağlam:
{context_text}

Soru: {question}

Cevap: /no_think"""
    return prompt

def main():
    print("Sistem hazırlanıyor...")
    collection=setup_retriever()
    chat_client=setup_llm()
    print("Hazır! Çıkmak için 'exit' yaz.\n")

    while True:
        question=input("Soru: ").strip()
        if question.lower() in ("exit","quit","çıkış"):
            break
        if not question:
            continue

        chunks,sources=retrieve_context(collection,question)

        if not chunks:
            print("İlgili bir bilgi bulunamadı.\n")
            continue

        prompt=build_prompt(question,chunks)

        full_answer=""
        for chunk in chat_client.complete_streaming_chat([
            {"role": "user","content": prompt}
        ]):
            if not chunk.choices:
                continue
            full_answer += chunk.choices[0].delta.content or ""

        clean_answer=re.sub(r"<think>.*?(</think>|$)","",full_answer,flags=re.DOTALL).strip()


        marker="**Cevap:**"
        if clean_answer.count(marker) > 1:
            first_end=clean_answer.find(marker,clean_answer.find(marker) + 1)
            clean_answer=clean_answer[:first_end].strip()

        if not clean_answer:
            clean_answer="(Model cevabı tamamlayamadı, lütfen tekrar deneyin.)"

        print(f"\nCevap: {clean_answer}")
        print(f"(Kaynaklar: {', '.join(set(sources))})\n")


if __name__ == "__main__":
    main()