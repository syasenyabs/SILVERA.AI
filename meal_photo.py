import sys
import os
import base64
import io

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from shared.llm_client import get_chat_client,unload_model,ask_image_via_responses_api
from shared.profile_store import get_profile

from PIL import Image

MODEL_ALIAS="qwen3-vl-8b-instruct"
MAX_DIMENSION=1024

MEDIA_TYPES={
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def encode_image_resized(path,max_dimension=MAX_DIMENSION):
    img=Image.open(path)
    if img.mode in ("RGBA","P"):
        img=img.convert("RGB")

    width,height=img.size
    if max(width,height) > max_dimension:
        if width > height:
            new_width=max_dimension
            new_height=int(height * (max_dimension / width))
        else:
            new_height=max_dimension
            new_width=int(width * (max_dimension / height))
        img=img.resize((new_width,new_height),Image.LANCZOS)
        print(f"   [bilgi] Görsel {width}x{height} -> {new_width}x{new_height} boyutuna küçültüldü.")

    buffer=io.BytesIO()
    img.save(buffer,format="JPEG",quality=85)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def bmi_category(bmi):
    if bmi < 18.5:
        return "zayıf"
    elif bmi < 25:
        return "normal aralık"
    elif bmi < 30:
        return "fazla kilolu"
    else:
        return "obez aralık"


def build_prompt(profile):
    profile_text=f"""Kullanıcı profili:
- Yaş: {profile['age']}
- Boy: {profile['height_cm']} cm
- Kilo: {profile['weight_kg']} kg
- BMI: {profile['bmi']} ({bmi_category(profile['bmi'])})
- Kronik rahatsızlık: {profile['chronic']}"""

    return f"""Sen bir diyetisyen asistanısın. Aşağıdaki fotoğrafta BİR ÖĞÜN
(genelde bir kase/tabak yiyecek VE varsa yanında bir içecek) var.
Bu fotoğrafı, kullanıcının profiline göre KİŞİSELLEŞTİRİLMİŞ şekilde analiz et.

{profile_text}

ÖNEMLİ - GÖRSELİ TAM TARA:
Fotoğrafta birden fazla öğe olabilir (örn. bir kase yiyecek VE ayrıca bir
fincan kahve/çay). Görselin TÜMÜNÜ (kenarları dahil) incele, sadece en
büyük/ortadaki öğeyi değil, TÜM yiyecek ve içecekleri tespit et.

ÖNEMLİ - TÜRKÇE İSİMLENDİRME:
Yiyecek isimlerini SADECE doğru, yaygın Türkçe adlarıyla yaz. Türkçe'ye
çevirirken hata yapma. Örnek doğru terimler:
- almond -> badem (ASLA "Alman" deme)
- peanut butter -> fıstık ezmesi (ASLA "Alman çorbası" deme)
- oatmeal / porridge -> yulaf lapası
- banana -> muz
- coffee -> kahve
Emin olmadığın bir yiyeceği "tanımlayamadığım bir besin" diye belirt,
uydurma bir isim verme.

SIRAYLA (bu sırayı değiştirme) şunları yap:
1. ÖNCE, bu öğünün (TÜM öğeleriyle - içecek dahil) kullanıcının profiline
   (BMI durumu ve varsa kronik rahatsızlığı) göre uygun olup olmadığını
   NET bir cümleyle söyle. Kronik rahatsızlık varsa MUTLAKA kısa, somut
   bir uyarı ekle.

2. SONRA fotoğraftaki TÜM yiyecek ve içecekleri (kahve/çay dahil) kısaca
   listele.

3. SON OLARAK (yer kalırsa) tahmini porsiyon büyüklüğü hakkında kısa bir not.

GÜVENLİK KURALI: Tanı koyma, kesin kalori sayısı iddia etme. Kronik
rahatsızlık varsa MUTLAKA "bu konuda diyetisyeninize/doktorunuza danışın" de.

Kısa ve net ol. Madde 1'i (kişiselleştirilmiş değerlendirme) MUTLAKA
tamamla, bu en önemli kısım. Toplamda en fazla 200 kelime kullan."""


def cut_duplicate_lines(text):
    lines=text.split("\n")
    result=[]
    prev_normalized=None

    for line in lines:
        normalized=line.strip().lower()
        if normalized and normalized == prev_normalized:
            continue
        result.append(line)
        prev_normalized=normalized

    return "\n".join(result)


def main():
    print("=== Öğün Fotoğrafı Analizi ===\n")

    name=input("İsminiz: ").strip()
    if not name:
        name="misafir"

    profile=get_profile(name)
    if not profile:
        print(f"\n'{name}' için kayıtlı bir profil bulunamadı.")
        print("Önce 'python modules/health.py' çalıştırıp profilini oluşturmalısın.")
        return

    print(f"\nHoş geldin, {profile.get('display_name',name)}! Profilin yüklendi.")
    print(f"  BMI: {profile['bmi']} ({bmi_category(profile['bmi'])}), Kronik: {profile['chronic']}\n")

    path=input("Fotoğrafın tam dosya yolunu gir (örn. C:\\Users\\...\\ogun.jpg): ").strip().strip('"')

    if not os.path.isfile(path):
        print("Dosya bulunamadı, yolu kontrol et.")
        return

    ext=os.path.splitext(path)[1].lower()
    if ext not in MEDIA_TYPES:
        print(f"Desteklenmeyen dosya türü: {ext}")
        return

    print("\nGörsel işleniyor ve model hazırlanıyor...")
    base64_image=encode_image_resized(path)
    prompt=build_prompt(profile)

    chat_client,model=get_chat_client(MODEL_ALIAS,max_tokens=1800,temperature=0.2)

    try:
        print("Analiz ediliyor (Responses API üzerinden), lütfen bekle...\n")
        answer=ask_image_via_responses_api(
            model,MODEL_ALIAS,prompt,base64_image,"image/jpeg",max_tokens=2500
        )

        answer=cut_duplicate_lines(answer)

        if answer and answer[-1] not in ".!?\"'”)":
            print("\n[UYARI] Cevap yarıda kesilmiş olabilir (token limiti dolmuş olabilir).\n")

        print("=== ANALİZ ===\n")
        print(answer)
    finally:
        unload_model(model)


if __name__ == "__main__":
    main()