# 🥗 Sağlık & Verimlilik Asistanı

Tamamen **yerelde (local-first)** çalışan, bulut API'sine ihtiyaç duymayan bir kişisel asistan uygulaması. [Foundry Local](https://github.com/microsoft/Foundry-Local) üzerinden çalıştırılan LLM'ler ile beslenme danışmanlığı, öğün fotoğrafı analizi ve görev planlaması yapar. Diyet Asistanı özelliği, doküman tabanlı bir **RAG (Retrieval-Augmented Generation)** mimarisi kullanır ve yalnızca sağlanan kaynak dokümanlardaki bilgilere dayanarak cevap üretir.

> ⚠️ **Not:** Bu proje tıbbi tavsiye vermez. Sunulan bilgiler genel bilgilendirme amaçlıdır; kronik rahatsızlıklar veya kesin beslenme kararları için mutlaka bir doktor/diyetisyene danışın.

---

## ✨ Özellikler

| Özellik | Açıklama |
|---|---|
| 👤 **Diyet Profili** | Yaş, boy, kilo, BMI ve kronik rahatsızlık bilgisini kaydeder; sonraki oturumlarda tekrar kullanılır. |
| 💬 **Diyet Asistanı (RAG)** | Kullanıcının sorusuna göre ilgili dokümanları (tip 1/tip 2 diyabet, Akdeniz diyeti vb.) ChromaDB üzerinden bulur, profile özel ve kaynağa dayalı cevap üretir. |
| 📸 **Öğün Analizi** | Yüklenen öğün fotoğrafını görsel-dil (vision) modeliyle analiz eder, kullanıcının profiline göre uygunluğunu değerlendirir. |
| 📅 **Görev Planlayıcı** | Rastgele sırada girilen görevleri, aralarındaki bağımlılık/eş zamanlılık ilişkisini çıkararak gerçekçi bir zaman planına dönüştürür. |

Uygulama hem **Streamlit tabanlı web arayüzü** (`app.py`) hem de bağımsız **terminal (CLI) script'leri** üzerinden kullanılabilir.

---

## 🧠 Kullanılan Modeller

Tüm modeller [Foundry Local](https://github.com/microsoft/Foundry-Local) ile cihaz üzerinde indirilip çalıştırılır, hiçbir veri dışarı gönderilmez.

| Model | Kullanım Alanı |
|---|---|
| `qwen3-8b` | Diyet Asistanı, genel RAG sorguları |
| `qwen3-vl-8b-instruct` | Öğün fotoğrafı analizi (görsel-dil modeli) |
| `phi-4` | Görev planlayıcı |

---

## 🛠️ Teknoloji Yığını

- **[Foundry Local SDK](https://github.com/microsoft/Foundry-Local)** — yerel model indirme/çalıştırma/sohbet
- **[Streamlit](https://streamlit.io/)** — web arayüzü
- **[ChromaDB](https://www.trychroma.com/)** — vektör veritabanı (RAG)
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) — embedding
- **pypdf**, **python-docx** — doküman okuma (PDF/DOCX/TXT)
- **Pillow** — görsel işleme
- **openai** (Python SDK) — Foundry Local'in OpenAI-uyumlu Responses API'si üzerinden görsel analiz

---

## 📁 Proje Yapısı

```
.
├── app.py                  # Streamlit ana uygulama (tüm sayfalar)
├── ingest.py                # Genel doküman -> ChromaDB ingest script'i (documents/ klasörü)
├── query.py                 # Bağımsız CLI RAG sorgu aracı
├── requirements.txt
├── documents/
│   └── diyet/                # Diyet Asistanı'nın kullandığı kaynak dokümanlar (PDF/DOCX/TXT)
├── shared/
│   ├── llm_client.py         # Foundry Local istemci sarmalayıcı (streaming chat, görsel analiz, vb.)
│   └── profile_store.py      # Kullanıcı profillerini profiles.json'da saklar
├── modules/
│   ├── health.py              # Diyet Asistanı mantığı (RAG, prompt, CLI modu)
│   ├── meal_photo.py          # Öğün fotoğrafı analizi (vision modeli, CLI modu)
│   └── planner.py             # Görev planlayıcı (CLI modu)
├── profiles.json             # Kullanıcı profilleri (otomatik oluşturulur)
├── chroma_db/                 # ingest.py / query.py için genel vektör veritabanı (otomatik oluşturulur)
└── chroma_db_diyet/           # Diyet Asistanı'na özel vektör veritabanı (otomatik oluşturulur)
```

---

## 🚀 Kurulum

### 1. Ön gereksinim: Foundry Local

Bu proje [Microsoft Foundry Local](https://github.com/microsoft/Foundry-Local)'ın cihazınızda kurulu ve çalışır durumda olmasını gerektirir. Kurulum talimatları için resmi depoyu inceleyin.

### 2. Depoyu klonlayın ve ortamı hazırlayın

```bash
git clone <bu-repo-url>
cd <proje-klasörü>
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 3. Diyet dokümanlarını ekleyin (opsiyonel ama Diyet Asistanı için gerekli)

`documents/diyet/` klasörüne PDF, DOCX veya TXT formatında kaynak dokümanlarınızı koyun. İlk çalıştırmada bu dokümanlar otomatik olarak işlenip `chroma_db_diyet/` içine indekslenir.

---

## ▶️ Kullanım

### Web arayüzü (önerilen)

```bash
streamlit run app.py
```

Tarayıcıda açılan arayüzden sırasıyla **Diyet Profili → Diyet Asistanı / Öğün Analizi / Görev Planlayıcı** sayfalarını kullanabilirsiniz.

### Bağımsız CLI script'leri

```bash
# Genel dokümanları (documents/ klasörü) vektör veritabanına ekler
python ingest.py

# Genel dokümanlar üzerinden terminalden soru-cevap
python query.py

# Diyet Asistanı (profil oluşturma + RAG sohbet), terminalden
python modules/health.py

# Öğün fotoğrafı analizi, terminalden
python modules/meal_photo.py

# Görev planlayıcı, terminalden
python modules/planner.py
```

---

## 🔒 Gizlilik

Uygulama `HF_HUB_OFFLINE=1` ve `TRANSFORMERS_OFFLINE=1` ortam değişkenlerini kullanarak dış ağ bağlantısı olmadan çalışacak şekilde ayarlanmıştır. Tüm model çıkarımı ve dosya işleme yerelde gerçekleşir; kullanıcı verileri (profil, fotoğraf, sorular) dışarıya gönderilmez.

---
---

# 🥗 Health & Productivity Assistant

A **fully local-first** personal assistant application that requires no cloud API. It uses LLMs served by [Foundry Local](https://github.com/microsoft/Foundry-Local) to provide nutrition guidance, meal photo analysis, and task planning. The Diet Assistant feature uses a document-based **RAG (Retrieval-Augmented Generation)** architecture and only answers based on the information contained in the provided source documents.

> ⚠️ **Note:** This project does not provide medical advice. The information given is for general informational purposes only; for chronic conditions or definitive dietary decisions, always consult a doctor/dietitian.

---

## ✨ Features

| Feature | Description |
|---|---|
| 👤 **Diet Profile** | Stores age, height, weight, BMI, and chronic condition info; reused across sessions. |
| 💬 **Diet Assistant (RAG)** | Finds relevant documents (type 1/type 2 diabetes, Mediterranean diet, etc.) via ChromaDB based on the user's question, producing profile-specific, source-grounded answers. |
| 📸 **Meal Analysis** | Analyzes an uploaded meal photo with a vision-language model and evaluates its suitability based on the user's profile. |
| 📅 **Task Planner** | Turns tasks entered in random order into a realistic time plan by inferring dependency/parallelism relationships between them. |

The app can be used via a **Streamlit web interface** (`app.py`) as well as standalone **terminal (CLI) scripts**.

---

## 🧠 Models Used

All models are downloaded and run on-device via [Foundry Local](https://github.com/microsoft/Foundry-Local); no data is sent externally.

| Model | Used For |
|---|---|
| `qwen3-8b` | Diet Assistant, general RAG queries |
| `qwen3-vl-8b-instruct` | Meal photo analysis (vision-language model) |
| `phi-4` | Task planner |

---

## 🛠️ Tech Stack

- **[Foundry Local SDK](https://github.com/microsoft/Foundry-Local)** — local model download/run/chat
- **[Streamlit](https://streamlit.io/)** — web interface
- **[ChromaDB](https://www.trychroma.com/)** — vector database (RAG)
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) — embeddings
- **pypdf**, **python-docx** — document reading (PDF/DOCX/TXT)
- **Pillow** — image processing
- **openai** (Python SDK) — used against Foundry Local's OpenAI-compatible Responses API for image analysis

---

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application (all pages)
├── ingest.py                # Generic document -> ChromaDB ingest script (documents/ folder)
├── query.py                 # Standalone CLI RAG query tool
├── requirements.txt
├── documents/
│   └── diyet/                # Source documents used by the Diet Assistant (PDF/DOCX/TXT)
├── shared/
│   ├── llm_client.py         # Foundry Local client wrapper (streaming chat, image analysis, etc.)
│   └── profile_store.py      # Stores user profiles in profiles.json
├── modules/
│   ├── health.py              # Diet Assistant logic (RAG, prompt, CLI mode)
│   ├── meal_photo.py          # Meal photo analysis (vision model, CLI mode)
│   └── planner.py             # Task planner (CLI mode)
├── profiles.json             # User profiles (auto-generated)
├── chroma_db/                 # Generic vector database for ingest.py / query.py (auto-generated)
└── chroma_db_diyet/           # Vector database dedicated to the Diet Assistant (auto-generated)
```

---

## 🚀 Setup

### 1. Prerequisite: Foundry Local

This project requires [Microsoft Foundry Local](https://github.com/microsoft/Foundry-Local) to be installed and running on your machine. See the official repo for installation instructions.

### 2. Clone the repo and set up the environment

```bash
git clone <this-repo-url>
cd <project-folder>
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 3. Add diet documents (optional, but required for the Diet Assistant)

Place your source documents (PDF, DOCX, or TXT) into the `documents/diyet/` folder. On first run, these documents are automatically processed and indexed into `chroma_db_diyet/`.

---

## ▶️ Usage

### Web interface (recommended)

```bash
streamlit run app.py
```

From the interface that opens in your browser, use the **Diet Profile → Diet Assistant / Meal Analysis / Task Planner** pages in order.

### Standalone CLI scripts

```bash
# Adds generic documents (documents/ folder) to the vector database
python ingest.py

# Terminal Q&A over the generic documents
python query.py

# Diet Assistant (profile creation + RAG chat), from the terminal
python modules/health.py

# Meal photo analysis, from the terminal
python modules/meal_photo.py

# Task planner, from the terminal
python modules/planner.py
```

---

## 🔒 Privacy

The app is configured to run without external network access using the `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` environment variables. All model inference and file processing happen locally; user data (profile, photos, questions) is never sent externally.
