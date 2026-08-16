import sys
import os


os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"

import re
import io
import time
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

import chromadb
from chromadb.utils import embedding_functions

from shared.llm_client import get_chat_client,unload_model,ask,ask_image_via_responses_api
from shared.profile_store import get_profile,save_profile,list_names

from modules import health as health_mod
from modules import meal_photo as meal_mod
from modules import planner as planner_mod


st.set_page_config(
    page_title="Sağlık & Verimlilik Asistanı",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    h1, h2, h3 { font-weight: 700; }

    .stButton>button {
        border-radius: 10px; font-weight: 600; padding: 0.5rem 1.2rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.18);
    }

    /* Kartlar kasıtlı olarak kendi rengini/arka planını taşıyor -
       böylece Streamlit light/dark tema ne olursa olsun okunabilir kalıyor. */
    .metric-card {
        background: linear-gradient(135deg, #eef2f9, #f8f9fb);
        border: 1px solid #dfe4ee; border-radius: 14px;
        padding: 1rem 1.2rem; text-align: center; color: #222;
        animation: fadeInUp 0.6s ease both;
        transition: transform 0.2s ease;
        min-height: 96px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-card:hover { transform: translateY(-4px); }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1a3c6e; }
    .metric-card .value-sub { font-size: 0.8rem; color: #7a8494; margin-top: 1px; min-height: 1.1rem; }
    .metric-card .label { font-size: 0.85rem; color: #5a6472; margin-top: 6px; }

    .warning-box {
        background: #fff8e6; border-left: 4px solid #f0ad4e; color: #6b4e00;
        padding: 0.8rem 1rem; border-radius: 8px; font-size: 0.9rem;
        margin-bottom: 1.2rem;
        animation: fadeIn 0.5s ease both;
    }

    .feature-card {
        background: linear-gradient(160deg, #ffffff, #f2f5fa);
        border: 1px solid #e3e7ee; border-radius: 16px;
        padding: 1.5rem 1.3rem; height: 100%; color: #222;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        animation: fadeInUp 0.6s ease both;
    }
    .feature-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 28px rgba(26,60,110,0.18);
    }
    .feature-card .icon {
        font-size: 1.9rem; margin-bottom: 0.5rem; display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    .feature-card h4 { margin: 0 0 0.4rem 0; color: #1a3c6e; }
    .feature-card p { margin: 0; font-size: 0.87rem; color: #555; line-height: 1.4; }

    .hero-title {
        font-size: 2.3rem; font-weight: 800; margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #1a3c6e, #4a8fd6, #1a3c6e);
        background-size: 200% auto;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: shine 5s linear infinite, fadeInUp 0.7s ease both;
    }
    .hero-sub {
        font-size: 1.02rem; opacity: 0.85;
        animation: fadeInUp 0.8s ease 0.1s both;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; } to { opacity: 1; }
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-4px); }
    }

    /* --- Kaydırmayla beliren (scroll-reveal) öğeler ---
       Varsayılan (her tarayıcı): sayfa yüklenince yumuşak fade-in.
       Destekleyen tarayıcılarda (Chrome/Edge 115+): gerçek kaydırma-
       tetiklemeli animasyona geçer - @supports bunu güvenli şekilde
       katmanlıyor, desteklemeyen tarayıcıda hiçbir şey bozulmaz. */
    .scroll-reveal {
        opacity: 0;
        animation: fadeInUp 0.7s ease forwards;
        animation-delay: var(--reveal-delay, 0s);
    }
    @supports (animation-timeline: view()) {
        .scroll-reveal {
            animation: revealScroll linear both;
            animation-timeline: view();
            animation-range: entry 0% cover 35%;
        }
    }
    @keyframes revealScroll {
        from { opacity: 0; transform: translateY(40px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    .section-title {
        font-size: 1.4rem; font-weight: 700; color: #1a3c6e;
        margin: 1.6rem 0 0.6rem 0;
    }

    .info-block {
        background: linear-gradient(160deg, #ffffff, #f2f5fa);
        border: 1px solid #e3e7ee; border-radius: 16px;
        padding: 1.4rem 1.5rem; margin-bottom: 1rem; color: #222;
    }
    .info-block h4 { margin: 0 0 0.4rem 0; color: #1a3c6e; display: flex; align-items: center; gap: 0.5rem; }
    .info-block p { margin: 0.3rem 0; font-size: 0.92rem; color: #444; line-height: 1.55; }
    .info-block ul { margin: 0.4rem 0 0 0; padding-left: 1.2rem; }
    .info-block li { font-size: 0.9rem; color: #444; margin-bottom: 0.25rem; }

    /* Kart altındaki "Aç" butonlarını karta bitişik/uyumlu göster */
    div[data-testid="stVerticalBlockBorderWrapper"] .stButton>button {
        width: 100%;
    }

    /* Sidebar karşılama kartı */
    .welcome-card {
        background: linear-gradient(135deg, #1a3c6e, #2d5a9e);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        color: white;
        animation: fadeInUp 0.6s ease both;
    }
    .welcome-card .greeting {
        font-size: 0.8rem;
        opacity: 0.85;
    }
    .welcome-card .name {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 2px;
    }
</style>
""",unsafe_allow_html=True)


def strip_think(text):
    return re.sub(r"<think>.*?(</think>|$)","",text or "",flags=re.DOTALL).strip()


def bmi_category(bmi):
    if bmi < 18.5:
        return "zayıf"
    elif bmi < 25:
        return "normal aralık"
    elif bmi < 30:
        return "fazla kilolu"
    else:
        return "obez aralık"


def get_greeting():
    hour=datetime.now().hour
    if hour < 6:
        return "İyi geceler"
    elif hour < 12:
        return "Günaydın"
    elif hour < 18:
        return "İyi günler"
    else:
        return "İyi akşamlar"


@st.cache_resource(show_spinner="Diyet dokümanları hazırlanıyor (ilk seferde biraz sürebilir)...")
def get_diet_collection():
    embedding_fn=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client=chromadb.PersistentClient(path=health_mod.DB_DIR)
    collection=client.get_or_create_collection(
        name="diyet_documents",embedding_function=embedding_fn
    )
    health_mod.ensure_ingested(collection)
    return collection


def run_llm(model_alias,prompt,max_tokens=1200,temperature=0.2,
            retries=2,backoff_seconds=8,spinner_text="Model çalışıyor..."):
    cache_key="_llm_cache"
    cached=st.session_state.get(cache_key)

    with st.spinner(spinner_text):
        if cached and cached["alias"] == model_alias:
            chat_client,model=cached["chat_client"],cached["model"]

            chat_client.settings.max_tokens=max_tokens
            chat_client.settings.temperature=temperature
        else:
            if cached:
                unload_model(cached["model"])
            chat_client,model=get_chat_client(
                model_alias,max_tokens=max_tokens,temperature=temperature
            )
            st.session_state[cache_key]={
                "alias": model_alias,"chat_client": chat_client,"model": model
            }

        answer=ask(chat_client,prompt,retries=retries,backoff_seconds=backoff_seconds)

    return strip_think(answer)


def ensure_profile_loaded():
    if "profile_name" not in st.session_state:
        st.session_state.profile_name=None
        st.session_state.profile_data=None


ensure_profile_loaded()

with st.sidebar:
    st.markdown("## 🥗 Sağlık & Verimlilik")
    st.caption("Local RAG tabanlı kişisel asistan")
    st.write("")

    PAGE_OPTIONS=["Ana Sayfa","Diyet Profili","Diyet Asistanı","Öğün Analizi","Görev Planlayıcı"]

    forced_index=st.session_state.pop("force_nav_index",None)

    page=option_menu(
        menu_title=None,
        options=PAGE_OPTIONS,
        icons=["house-door","person-vcard","chat-left-dots","camera","calendar-check"],
        default_index=0,
        manual_select=forced_index,
        key="main_nav_menu",
        styles={
            "container": {"padding": "6px!important","background-color": "#f4f6fa","border-radius": "14px"},
            "icon": {"font-size": "17px"},
            "nav-link": {
                "font-size": "15px",
                "font-weight": "500",
                "text-align": "left",
                "margin": "3px 0",
                "border-radius": "10px",
                "padding": "10px 14px",
                "color": "#333333",
                "--hover-color": "#e3e8f2",
                "transition": "all 0.2s ease",
            },
            "nav-link-selected": {
                "background-color": "#1a3c6e",
                "color": "white",
                "font-weight": "600",
            },
        },
    )

    st.write("")
    st.divider()

    if st.session_state.profile_data:
        p=st.session_state.profile_data
        display_name=p.get("display_name",st.session_state.profile_name)

        st.markdown(
            f'<div class="welcome-card">'
            f'<div class="greeting">{get_greeting()} 👋</div>'
            f'<div class="name">{display_name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"BMI: {p['bmi']} ({bmi_category(p['bmi'])}) · "
            f"Kronik: {p.get('chronic','belirtilmedi')}"
        )
    else:
        st.info("Henüz profil yüklenmedi. **Diyet Profili** sayfasından oluştur.")


def page_home():
    st.markdown('<div class="hero-title">🥗 Sağlık & Verimlilik Asistanı</div>',unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Kişisel bilgilerine göre beslenme tavsiyesi al, öğün fotoğrafını '
        'analiz ettir ve görevlerini planla — hepsi tamamen yerelde (local) çalışır.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("")


    features=[
        (
            "💬","Diyet Asistanı",
            "Dokümanlara dayalı, kişiselleştirilmiş beslenme soruları.",
            "Diyet Asistanı'nı Aç →",2,
            "Yerel dokümanlardaki (kan grubu, kronik hastalıklar, tip 1/tip 2 diyabet, "
            "cinsiyete göre beslenme vb.) bilgilere dayanarak, profiline özel ve kaynaklı "
            "cevaplar üreten bir RAG (Retrieval-Augmented Generation) sohbet asistanı.",
            [
                "Sorunu yazınca ilgili dokümanları otomatik bulur ve ona göre cevaplar",
                "Kan grubuna, kronik rahatsızlığına göre otomatik filtreleme yapar",
                "Cevaplar sadece dokümanlardaki gerçek bilgiye dayanır, uydurma yapmaz",
            ],
        ),
        (
            "📸","Öğün Analizi",
            "Fotoğraf yükle, profiline göre kişiselleştirilmiş değerlendirme al.",
            "Öğün Analizi'ni Aç →",3,
            "Bir öğün fotoğrafı yükle; görsel analiz modeli tabaktaki yiyecekleri tanır "
            "ve senin yaş, BMI ve kronik rahatsızlık bilgine göre bu öğünün ne kadar "
            "uygun olduğunu değerlendirip kişisel bir geri bildirim verir.",
            [
                "Fotoğraftaki yiyecekleri ve tahmini porsiyonları listeler",
                "Öğünün profiline göre uygunluğunu (örn. tip 1 diyabet için) değerlendirir",
                "Kronik rahatsızlık varsa mutlaka doktor/diyetisyene danışma uyarısı ekler",
            ],
        ),
        (
            "📅","Görev Planlayıcı",
            "Görevlerini gir, model senin için doğru sırayı ve planı çıkarsın.",
            "Görev Planlayıcı'yı Aç →",4,
            "Yapman gereken görevleri rastgele sırayla gir, süre/açıklama ekle - model "
            "hangi görevlerin birbirine bağımlı, hangilerinin eş zamanlı yapılabileceğini "
            "analiz edip senin için gerçekçi bir zaman planı çıkarır.",
            [
                "Görevleri istediğin sırada girebilirsin, doğru sırayı model bulur",
                "Eş zamanlı yapılabilecek görevleri (örn. 'video izlerken not al') fark eder",
                "Belirttiğin toplam zaman dilimine göre somut bir plan sunar",
            ],
        ),
    ]

    cols=st.columns(3)
    for i,(col,(icon,title,short_desc,btn_label,target_index,_,_)) in enumerate(zip(cols,features)):
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div class="feature-card scroll-reveal" style="--reveal-delay:{i * 0.12}s; '
                    f'border:none; margin:-1px -1px 0.6rem -1px; padding:0.2rem 0.1rem 0 0.1rem;">'
                    f'<span class="icon">{icon}</span>'
                    f'<h4>{title}</h4><p>{short_desc}</p></div>',
                    unsafe_allow_html=True,
                )
                if st.button(btn_label,key=f"home_btn_{i}",use_container_width=True):
                    st.session_state.force_nav_index=target_index
                    st.rerun()

    st.write("")
    st.divider()

    if not st.session_state.profile_data:
        st.markdown(
            '<div class="warning-box">👤 Kişiselleştirilmiş sonuçlar için önce '
            '<b>Diyet Profili</b> sayfasından bilgilerini gir.</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Diyet Profilini Oluştur →",key="home_profile_btn"):
            st.session_state.force_nav_index=1
            st.rerun()
    else:
        p=st.session_state.profile_data
        c1,c2,c3,c4=st.columns(4)
        items=[
            ("Yaş",str(p["age"]),None),
            ("BMI",str(p["bmi"]),bmi_category(p["bmi"])),
            ("Durum","Aktif profil",None),
            ("Kronik Rahatsızlık",p.get("chronic","belirtilmedi"),None),
        ]
        for i,(col,(label,value,sub)) in enumerate(zip([c1,c2,c3,c4],items)):
            with col:
                sub_html=f'<div class="value-sub">{sub or "&nbsp;"}</div>'
                st.markdown(
                    f'<div class="metric-card scroll-reveal" style="--reveal-delay:{i * 0.1}s">'
                    f'<div class="value">{value}</div>'
                    f'{sub_html}'
                    f'<div class="label">{label}</div></div>',
                    unsafe_allow_html=True,
                )


    st.write("")
    st.write("")
    st.markdown('<div class="section-title scroll-reveal">🔍 Bu araçlar tam olarak ne işe yarar?</div>',unsafe_allow_html=True)

    for i,(icon,title,_,_,_,detail,bullets) in enumerate(features):
        bullets_html="".join(f"<li>{b}</li>" for b in bullets)
        st.markdown(
            f'<div class="info-block scroll-reveal" style="--reveal-delay:{i * 0.08}s">'
            f'<h4>{icon} {title}</h4>'
            f'<p>{detail}</p>'
            f'<ul>{bullets_html}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )


def page_profile():
    st.title("👤 Diyet Profili")

    existing_names=list_names()
    if existing_names:
        st.caption("Kayıtlı bir profili yükle veya aşağıdan yeni profil oluştur.")
        chosen=st.selectbox("Kayıtlı profiller",["-- seçim yok --"] + existing_names)
        if chosen != "-- seçim yok --":
            if st.button("Bu profili yükle"):
                key_guess=chosen.strip().lower()
                loaded=get_profile(key_guess)
                if loaded:
                    st.session_state.profile_name=key_guess
                    st.session_state.profile_data=loaded
                    st.success(f"'{chosen}' profili yüklendi.")
                    st.rerun()

    st.divider()
    st.subheader("Yeni profil oluştur / güncelle")

    with st.form("profile_form"):
        name=st.text_input("İsim",value=st.session_state.get("profile_name","") or "")
        col1,col2,col3=st.columns(3)
        with col1:
            age=st.number_input("Yaş",min_value=1,max_value=120,value=25)
        with col2:
            height_cm=st.number_input("Boy (cm)",min_value=50.0,max_value=250.0,value=170.0)
        with col3:
            weight_kg=st.number_input("Kilo (kg)",min_value=20.0,max_value=300.0,value=70.0)
        chronic=st.text_input("Kronik rahatsızlık (yoksa boş bırak)",value="")

        submitted=st.form_submit_button("Kaydet")

        if submitted:
            if not name.strip():
                st.error("Lütfen bir isim gir.")
            else:
                height_m=height_cm / 100
                bmi=round(weight_kg / (height_m ** 2),1)
                profile={
                    "age": int(age),
                    "height_cm": height_cm,
                    "weight_kg": weight_kg,
                    "bmi": bmi,
                    "chronic": chronic.strip() if chronic.strip() else "belirtilmedi",
                }
                save_profile(name,profile)
                profile["display_name"]=name.strip()
                st.session_state.profile_name=name.strip().lower()
                st.session_state.profile_data=profile
                st.success(f"Profil kaydedildi. BMI: {bmi} ({bmi_category(bmi)})")
                st.rerun()


def page_diet_assistant():
    st.title("💬 Diyet Asistanı")

    if not st.session_state.profile_data:
        st.warning("Önce **Diyet Profili** sayfasından bilgilerini gir.")
        return

    profile=st.session_state.profile_data
    collection=get_diet_collection()

    if profile.get("chronic","belirtilmedi") != "belirtilmedi":
        st.caption(
            "ℹ️ Kronik rahatsızlığınla ilgili öneriler genel bilgilendirme amaçlıdır, "
            "kesin karar için doktoruna/diyetisyenine danış."
        )

    if "diet_chat_history" not in st.session_state:
        st.session_state.diet_chat_history=[]

    for role,content in st.session_state.diet_chat_history:
        with st.chat_message(role):
            st.write(content)

    question=st.chat_input("Beslenme ile ilgili bir soru sor...")

    if question:
        st.session_state.diet_chat_history.append(("user",question))
        with st.chat_message("user"):
            st.write(question)

        target_source=health_mod.find_target_source(question,profile)

        if target_source:
            results=collection.query(
                query_texts=[question],n_results=health_mod.TOP_K,
                where={"source": target_source},
            )
        else:
            results=collection.query(query_texts=[question],n_results=health_mod.TOP_K)

        all_chunks=results["documents"][0]
        all_distances=results["distances"][0]
        chunks=[c for c,d in zip(all_chunks,all_distances) if d <= health_mod.DISTANCE_THRESHOLD]

        prompt=health_mod.build_prompt(profile,chunks,question)

        answer=run_llm(
            health_mod.MODEL_ALIAS,prompt,max_tokens=1800,temperature=0.15,
            spinner_text="Cevap hazırlanıyor...",
        )
        answer=health_mod.cut_repetition(answer)
        answer=health_mod.limit_numbered_items(answer,max_items=5)

        st.session_state.diet_chat_history.append(("assistant",answer))
        with st.chat_message("assistant"):
            st.write(answer)


def page_meal_analysis():
    st.title("📸 Öğün Fotoğrafı Analizi")

    if not st.session_state.profile_data:
        st.warning("Önce **Diyet Profili** sayfasından bilgilerini gir.")
        return

    profile=st.session_state.profile_data

    uploaded=st.file_uploader("Öğün fotoğrafını yükle",type=["jpg","jpeg","png","webp"])

    if uploaded:
        img=Image.open(uploaded)
        st.image(img,caption="Yüklenen fotoğraf",width=350)

        if st.button("Analiz et"):
            if img.mode in ("RGBA","P"):
                img=img.convert("RGB")

            width,height=img.size
            max_dim=meal_mod.MAX_DIMENSION
            if max(width,height) > max_dim:
                if width > height:
                    new_w,new_h=max_dim,int(height * (max_dim / width))
                else:
                    new_h,new_w=max_dim,int(width * (max_dim / height))
                img=img.resize((new_w,new_h),Image.LANCZOS)

            buf=io.BytesIO()
            img.save(buf,format="JPEG",quality=85)
            buf.seek(0)
            import base64
            base64_image=base64.b64encode(buf.read()).decode("utf-8")

            prompt=meal_mod.build_prompt(profile)

            with st.spinner("Fotoğraf analiz ediliyor, bu biraz sürebilir..."):
                chat_client,model=get_chat_client(
                    meal_mod.MODEL_ALIAS,max_tokens=1800,temperature=0.2
                )
                try:
                    answer=ask_image_via_responses_api(
                        model,meal_mod.MODEL_ALIAS,prompt,base64_image,
                        "image/jpeg",max_tokens=1800,
                    )
                finally:
                    unload_model(model)

            answer=strip_think(answer)

            st.divider()
            st.subheader("Analiz")
            st.write(answer)


def page_planner():
    st.title("📅 Görev Planlayıcı")
    st.caption("Görevlerini rastgele sırayla girebilirsin, model doğru sırayı/planı bulur.")

    if "planner_tasks" not in st.session_state:
        st.session_state.planner_tasks=[{"task": "","duration": ""}]

    for i,item in enumerate(st.session_state.planner_tasks):
        col1,col2=st.columns([2,1])
        with col1:
            item["task"]=st.text_input(f"Görev {i + 1}",value=item["task"],key=f"task_{i}")
        with col2:
            item["duration"]=st.text_input(
                "Süre / açıklama",value=item["duration"],key=f"dur_{i}",
                placeholder="örn. 2 saat"
            )

    col_add,col_remove=st.columns(2)
    with col_add:
        if st.button("➕ Görev ekle"):
            st.session_state.planner_tasks.append({"task": "","duration": ""})
            st.rerun()
    with col_remove:
        if len(st.session_state.planner_tasks) > 1 and st.button("➖ Son görevi sil"):
            st.session_state.planner_tasks.pop()
            st.rerun()

    timeframe=st.text_input("Toplam zaman dilimi (örn. '1 hafta')",value="")

    if st.button("📋 Planı oluştur",type="primary"):
        tasks=[t["task"].strip() for t in st.session_state.planner_tasks if t["task"].strip()]
        durations=[
            t["duration"].strip() or "belirtilmedi"
            for t in st.session_state.planner_tasks if t["task"].strip()
        ]

        if not tasks:
            st.error("En az bir görev girmelisin.")
            return

        tasks_text="\n".join(f"- {t} ({d})" for t,d in zip(tasks,durations))
        tf=timeframe.strip() or "belirtilmedi, makul bir süre varsay"

        prompt=planner_mod.build_prompt(tasks_text,tf)

        answer=run_llm(
            planner_mod.MODEL_ALIAS,prompt,max_tokens=900,temperature=0.2,
            spinner_text="Plan hazırlanıyor (model adım adım düşünüyor)...",
        )

        st.divider()
        st.subheader("Plan")
        st.write(answer)


if page == "Ana Sayfa":
    page_home()
elif page == "Diyet Profili":
    page_profile()
elif page == "Diyet Asistanı":
    page_diet_assistant()
elif page == "Öğün Analizi":
    page_meal_analysis()
elif page == "Görev Planlayıcı":
    page_planner()