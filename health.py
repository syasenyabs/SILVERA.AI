import sys
import os

os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"

import re
import io
import time
from datetime import datetime


sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from shared.llm_client import get_chat_client,unload_model,ask
from shared.profile_store import get_profile,save_profile

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import docx

DOCUMENTS_DIR=os.path.join("documents","diyet")
DB_DIR="chroma_db_diyet"
MODEL_ALIAS="qwen3-8b"
TOP_K=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DISTANCE_THRESHOLD=1.1


KEYWORD_SOURCE_MAP={
    "tip 1 diyabet": "tip1diyabet.pdf",
    "tip1 diyabet": "tip1diyabet.pdf",
    "tip 2 diyabet": "tip2diyabet.pdf",
    "tip2 diyabet": "tip2diyabet.pdf",
    "diyabet": "tip2diyabet.pdf",
    "pkos": "pcosbeslenme.pdf",
    "pcos": "pcosbeslenme.pdf",
    "polikistik": "pcosbeslenme.pdf",
    "kan grubu": "kangrubunagörebeslenmee.pdf",
    "kan grubuna": "kangrubunagörebeslenmee.pdf",
    "akdeniz": "akdenizdiyeti.pdf",
}


def find_target_source(question,profile=None):
    q=question.lower()
    for keyword,source in KEYWORD_SOURCE_MAP.items():
        if keyword in q:
            return source

    if profile:
        chronic=str(profile.get("chronic","")).lower()
        for keyword,source in KEYWORD_SOURCE_MAP.items():
            if keyword in chronic:
                return source

    return None


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


def ensure_ingested(collection):
    if collection.count() > 0:
        return

    if not os.path.isdir(DOCUMENTS_DIR):
        print(f"UYARI: '{DOCUMENTS_DIR}' klasörü bulunamadı, boş çalışılacak.")
        return

    files=[f for f in os.listdir(DOCUMENTS_DIR) if not f.startswith(".")]
    if not files:
        print(f"UYARI: '{DOCUMENTS_DIR}' klasörü boş.")
        return

    print("Diyet dokümanları ilk kez işleniyor (bu sadece bir kereye mahsus)...")
    for filename in files:
        path=os.path.join(DOCUMENTS_DIR,filename)
        text=load_document(path)
        if not text:
            continue
        chunks=chunk_text(text)
        if not chunks:
            continue
        ids=[f"{filename}_{i}" for i in range(len(chunks))]
        metadatas=[{"source": filename,"chunk_index": i} for i in range(len(chunks))]
        collection.upsert(documents=chunks,ids=ids,metadatas=metadatas)
        print(f"  {filename}: {len(chunks)} parça eklendi.")
    print("Doküman işleme tamamlandı.\n")


def cut_repetition(text,min_repeats=2):
    def cut_inline_repetition(line,min_repeats=3):
        parts=re.split(r",\s*",line)
        seen_counts={}
        result_parts=[]
        for part in parts:
            normalized=part.strip().lower()
            if len(normalized) < 2:
                result_parts.append(part)
                continue
            seen_counts[normalized]=seen_counts.get(normalized,0) + 1
            if seen_counts[normalized] >= min_repeats:
                break
            result_parts.append(part)
        return ", ".join(result_parts)

    lines=text.split("\n")
    seen_counts={}
    result_lines=[]
    for line in lines:
        line=cut_inline_repetition(line)

        normalized=re.sub(r"^\s*\d+[\.\)]\s*","",line.strip().lower())
        if len(normalized) < 15:
            result_lines.append(line)
            continue
        seen_counts[normalized]=seen_counts.get(normalized,0) + 1
        if seen_counts[normalized] >= min_repeats:
            break
        result_lines.append(line)
    return "\n".join(result_lines).strip()

def limit_numbered_items(text,max_items=5):
    lines=text.split("\n")
    result_lines=[]
    item_count=0
    for line in lines:
        match=re.match(r"^\s*(\d+)[\.\)]\s",line)
        if match:
            item_count += 1
            if item_count > max_items:
                break
        result_lines.append(line)
    return "\n".join(result_lines).strip()


def bmi_category(bmi):
    if bmi < 18.5:
        return "zayıf"
    elif bmi < 25:
        return "normal aralık"
    elif bmi < 30:
        return "fazla kilolu"
    else:
        return "obez aralık"


def collect_new_profile(name):
    print(f"\n'{name}' için yeni profil oluşturuluyor.")
    while True:
        try:
            age=int(input("Yaş: ").strip())
            break
        except ValueError:
            print("Lütfen sayı olarak gir.")

    while True:
        try:
            height_cm=float(input("Boy (cm): ").strip())
            break
        except ValueError:
            print("Lütfen sayı olarak gir.")

    while True:
        try:
            weight_kg=float(input("Kilo (kg): ").strip())
            break
        except ValueError:
            print("Lütfen sayı olarak gir.")

    chronic=input("Kronik bir rahatsızlık var mı? (yoksa boş bırakın): ").strip()

    height_m=height_cm / 100
    bmi=round(weight_kg / (height_m ** 2),1)

    profile={
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "chronic": chronic if chronic else "belirtilmedi"
    }
    save_profile(name,profile)
    return profile


def get_or_create_profile():
    name=input("İsminiz: ").strip()
    if not name:
        name="misafir"

    existing=get_profile(name)
    if existing:
        print(f"\nHoş geldin, {existing.get('display_name',name)}! Kayıtlı profilin bulundu:")
        print(f"  Yaş: {existing['age']}, Boy: {existing['height_cm']} cm, Kilo: {existing['weight_kg']} kg, "
              f"BMI: {existing['bmi']} ({bmi_category(existing['bmi'])}), Kronik: {existing['chronic']}")
        update=input("Bilgilerini güncellemek ister misin? (evet/hayır): ").strip().lower()
        if update in ("evet","e","yes","y"):
            return name,collect_new_profile(name)
        return name,existing
    else:
        print(f"\n'{name}' için kayıtlı profil bulunamadı, yeni profil oluşturalım.")
        return name,collect_new_profile(name)


def build_prompt(profile,context_chunks,question):
    context_text="\n\n---\n\n".join(context_chunks) if context_chunks else "(İlgili doküman bulunamadı)"

    prompt=f"""Sen bir diyetisyen asistanısın. Aşağıdaki kullanıcı profilini ve doküman bağlamını kullanarak soruyu cevapla.

KULLANICI PROFİLİ:
- Yaş: {profile['age']}
- Boy: {profile['height_cm']} cm
- Kilo: {profile['weight_kg']} kg
- BMI: {profile['bmi']} ({bmi_category(profile['bmi'])})
- Kronik rahatsızlık: {profile['chronic']}

DOKÜMAN BAĞLAMI:
{context_text}

GÜVENLİK VE DOĞRULUK KURALLARI (MUTLAKA UY):
- Sen bir doktor değilsin, TANI KOYMA, kesin tedavi önerme.
- SADECE yukarıdaki DOKÜMAN BAĞLAMI'nda geçen bilgileri kullan. Doküman bağlamında olmayan bir iddiayı EKLEME.
- Eğer doküman bağlamı soruyla ilgisizse veya boşsa, kendi genel bilgini KULLANMA - sadece "bu konuda elimdeki dokümanlarda yeterli bilgi yok, bir diyetisyene danışmanız önerilir" de.
- Kronik bir rahatsızlık belirtilmişse, MUTLAKA "bu konuda diyetisyeninize/doktorunuza danışın" de.
- UZUNLUK SINIRI: Cevabın EN FAZLA 250 kelime olsun. Madde listesi yapıyorsan EN FAZLA 5 madde yaz, her madde EN FAZLA 1 cümle olsun - açıklama kısmını her maddede tekrarlama.
- ASLA aynı cümleyi veya maddeyi tekrar yazma. Her satır/madde SADECE BİR KEZ geçmeli. Bir şeyi yazdıktan sonra bir daha aynı şeyi tekrarlama.

EĞER BİR ÖĞÜN PLANI İSTENİYORSA, ŞU İKİ ADIMI TAKİP ET:
ADIM 1 - BESİN GRUBU DAĞILIMI (kısaca, tek satırda göster):
Her öğüne FARKLI bir ana besin grubu ata, aynı grubu iki öğünde kullanma. Örnek: "Kahvaltı: yumurta+süt grubu, Öğle: et/protein ağırlıklı, Akşam: baklagil/sebze ağırlıklı, Ara öğün: meyve/kuruyemiş"

ADIM 2 - ÖĞÜN LİSTESİ:
Adım 1'deki dağılıma göre, her öğün için TAM OLARAK 2 madde halinde SOMUT yiyecek yaz (fazla açıklama ekleme, sadece besin adı + kısa not). Hiçbir öğün diğeriyle aynı olmasın. Fazladan "Notlar"/"Sonuç" bölümü YAZMA.
Soru: {question}
ADIM 2 - ÖĞÜN LİSTESİ:
Adım 1'deki dağılıma göre, her öğün için TAM OLARAK 2 madde halinde SOMUT yiyecek yaz (fazla açıklama ekleme, sadece besin adı + kısa not). Hiçbir öğün diğeriyle aynı olmasın. Fazladan "Notlar"/"Sonuç" bölümü YAZMA.ADIM 2 - ÖĞÜN LİSTESİ:
Cevap: /no_think"""
    return prompt


def main():
    print("=== Diyet Asistanı ===\n")

    name,profile=get_or_create_profile()

    embedding_fn=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client=chromadb.PersistentClient(path=DB_DIR)
    collection=client.get_or_create_collection(
        name="diyet_documents",
        embedding_function=embedding_fn
    )
    ensure_ingested(collection)

    if profile["chronic"] != "belirtilmedi":
        print("\nNot: Kronik rahatsızlığınla ilgili öneriler genel bilgilendirme amaçlıdır, kesin karar için doktoruna/diyetisyenine danış.")

    chat_client,model=get_chat_client(MODEL_ALIAS,max_tokens=1450,temperature=0.15)

    try:
        print("\nHazır! Beslenme ile ilgili sorularını sorabilirsin. Çıkmak için 'exit' yaz.\n")
        while True:
            question=input("Soru: ").strip()
            if question.lower() in ("exit","quit","çıkış"):
                break
            if not question:
                continue

            target_source=find_target_source(question)
            if target_source:
                results=collection.query(
                    query_texts=[question],
                    n_results=TOP_K,
                    where={"source": target_source}
                )
            else:
                results=collection.query(query_texts=[question],n_results=TOP_K)

            all_chunks=results["documents"][0]
            all_distances=results["distances"][0]


            chunks=[c for c,d in zip(all_chunks,all_distances) if d <= DISTANCE_THRESHOLD]

            prompt=build_prompt(profile,chunks,question)
            answer=ask(chat_client,prompt,max_retries=2)
            answer=re.sub(r"<think>.*?(</think>|$)","",answer,flags=re.DOTALL).strip()
            answer=cut_repetition(answer)
            answer=limit_numbered_items(answer,max_items=5)
            print(f"\nCevap: {answer}\n")
    finally:
        unload_model(model)


if __name__ == "__main__":
    main()