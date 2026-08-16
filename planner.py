import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from shared.llm_client import get_chat_client,unload_model,ask

MODEL_ALIAS="phi-4"


def build_prompt(tasks_text,timeframe):
    prompt=f"""Aşağıda bir kullanıcının yapması gereken görevler var. Görevler RASTGELE SIRAYLA verilmiştir, doğru sırayı SEN bulmalısın. Her görevin yanında kullanıcının belirttiği tahmini süre/açıklama da var - bunları MUTLAKA kullan, kendi tahminini yapma. Görev listesi tam olarak {tasks_text.count(chr(10)) + 1} maddeden oluşur, bundan fazla veya az görev UYDURMA.

Görevler (süre/açıklamalarıyla birlikte):
{tasks_text}

Zaman dilimi: {timeframe}

ÖNEMLİ KURAL - EŞ ZAMANLI (PARALEL) GÖREVLER:
Bazı görevler birbirine BAĞIMLI değil, birbiriyle EŞ ZAMANLI (aynı anda/birlikte) yapılabilir. Açıklamada "X yaparken Y de yapılır", "X ile birlikte", "aynı anda" gibi ifadeler varsa, bu görevleri SIRALI değil, PARALEL planla.

ÖRNEK 1 (bağımlılık - sıralı olması gereken durum):
Görevler: "Rapor yaz (2 gün)", "Deney yap (1 gün)", "Sunum hazırla (1 gün)"
Doğru sıra: 1) Deney yap (1 gün), 2) Rapor yaz (2 gün), 3) Sunum hazırla (1 gün)

ÖRNEK 2 (eş zamanlı/paralel durum):
Görevler: "Video izle (uzun sürer)", "Not çıkar (video izlerken yapılır)"
Doğru plan: Video izle VE Not çıkar aynı gün(ler)de birlikte gösterilir.

Şimdi sıra sende. Şu adımları TAKİP ET ve cevabında GÖSTER:
ADIM 1 - GÖREV SAYISI KONTROLÜ:
Yukarıda kaç görev verildiğini say ve listele (görev metnini değiştirmeden, olduğu gibi). Bu adım, hiçbir görevi atlamadığından veya uydurmadığından emin olmak içindir.

ADIM 2 - BAĞIMLILIK VE EŞ ZAMANLILIK ANALİZİ:
Her görev için, (a) hangi görev(ler)den SONRA yapılması gerektiğini (bağımlılık), YA DA (b) hangi görev(ler)le AYNI ANDA/PARALEL yapılabileceğini (eş zamanlılık) belirle. Bunu kısaca yaz.

ADIM 3 - DOĞRU SIRA/GRUPLAMA:
Bağımlı görevleri sıralı, eş zamanlı görevleri aynı grupta göster.

ADIM 4 - PLAN:
Verilen süreleri KULLANARAK, zaman dilimine göre somut bir plan çıkar. Eğer toplam süre zaman diliminden fazlaysa bunu açıkça belirt.

Kısa ve net ol."""
    return prompt


def main():
    print("=== Görev Planlayıcı ===")
    print("Görevlerini yaz (istediğin sırayla girebilirsin, model doğru sırayı kendi bulacak).")
    print("Her görevden sonra Enter'a bas. Bitirince boş satır bırakıp Enter'a bas.\n")

    tasks=[]
    while True:
        line=input(f"Görev {len(tasks) + 1}: ").strip()
        if not line:
            break
        tasks.append(line)

    if not tasks:
        print("Hiç görev girmedin, çıkılıyor.")
        return

    print("\nŞimdi her görev için tahmini süreyi/açıklamayı gir.")
    print("(Eş zamanlı yapılabilecek bir görevse, bunu belirt - örn. 'video izlerken yapılır')")
    print("(Kısa ve net yaz, virgülle birden fazla bilgiyi birleştirme)\n")
    durations=[]
    for t in tasks:
        d=input(f"'{t}' için süre/açıklama: ").strip()
        if not d:
            d="belirtilmedi"
        durations.append(d)

    timeframe=input("\nBu görevler için toplam zaman dilimin nedir? (örn. '1 hafta'): ").strip()
    if not timeframe:
        timeframe="belirtilmedi, makul bir süre varsay"


    print(f"\n--- Girdiğin {len(tasks)} görev şu şekilde anlaşıldı ---")
    for i,(t,d) in enumerate(zip(tasks,durations),start=1):
        print(f"{i}. Görev: \"{t}\"  |  Açıklama: \"{d}\"")

    confirm=input("\nBu liste doğru mu? (evet/hayır): ").strip().lower()
    if confirm not in ("evet","e","yes","y"):
        print("Lütfen görevleri/açıklamaları düzelterek tekrar başlat.")
        return

    tasks_text="\n".join(f"- {t} ({d})" for t,d in zip(tasks,durations))
    prompt=build_prompt(tasks_text,timeframe)

    print("\nPlan hazırlanıyor, bu biraz zaman alabilir (model adım adım düşünüyor)...\n")
    chat_client,model=get_chat_client(MODEL_ALIAS,max_tokens=900,temperature=0.2)

    try:
        answer=ask(chat_client,prompt)
        print("=== PLAN ===\n")
        print(answer)
    finally:
        unload_model(model)


if __name__=="__main__":
    main()