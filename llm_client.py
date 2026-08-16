from foundry_local_sdk import Configuration,FoundryLocalManager
import subprocess
import re

_manager_initialized=False


def _get_endpoint_via_cli():
    try:
        result=subprocess.run(
            ["foundry","status"],
            capture_output=True,text=True,timeout=15,
            encoding="utf-8",errors="ignore",shell=True
        )
        output=result.stdout
        print(f"   [debug] foundry status çıktısı (ilk 300 kr): {repr(output[:300])}")
        match=re.search(r"(https?://[\w\.\-]+:\d+)",output)
        if match:
            return match.group(1)
        print("   [debug] Web URL deseni çıktıda bulunamadı.")
    except Exception as e:
        print(f"   [debug] 'foundry status' çalıştırılamadı: {e}")
    return None


def _ensure_manager():
    global _manager_initialized
    if not _manager_initialized:
        FoundryLocalManager.initialize(Configuration(app_name="ai-assistant-project"))
        _manager_initialized=True


def get_chat_client(model_alias,max_tokens=1100,temperature=0.2,top_p=0.9):
    _ensure_manager()
    manager=FoundryLocalManager.instance
    model=manager.catalog.get_model(model_alias)
    model.download()
    model.load()

    chat_client=model.get_chat_client()
    chat_client.settings.max_tokens=max_tokens
    chat_client.settings.temperature=temperature
    chat_client.settings.top_p=top_p

    return chat_client,model


def unload_model(model):
    try:
        model.unload()
    except Exception:
        pass


def ask(chat_client,prompt,retries=1):
    attempt=0
    while True:
        try:
            full_answer=""
            for chunk in chat_client.complete_streaming_chat([
                {"role": "user","content": prompt}
            ]):
                if not chunk.choices:
                    continue
                full_answer += chunk.choices[0].delta.content or ""
            return full_answer.strip()

        except Exception as e:
            if attempt < retries:
                attempt += 1
                print(
                    f"   [uyarı] Geçici bir hata oluştu, tekrar "
                    f"deneniyor... ({e})"
                )
                continue

            raise


def ask_with_history(chat_client,messages):
    full_answer=""
    for chunk in chat_client.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        full_answer += chunk.choices[0].delta.content or ""
    return full_answer.strip()


def ask_with_image(chat_client,prompt_text,base64_image,media_type="image/jpeg"):
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text","text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{base64_image}"}
                }
            ]
        }
    ]
    full_answer=""
    for chunk in chat_client.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        full_answer += chunk.choices[0].delta.content or ""
    return full_answer.strip()


def ask_image_via_responses_api(
    model,
    model_alias,
    prompt_text,
    base64_image,
    media_type="image/jpeg",
    max_tokens=1800
):
    import openai

    _ensure_manager()
    manager=FoundryLocalManager.instance

    endpoint=None
    api_key="not-needed"
    for attr in ("endpoint","service_url","web_service_url","base_url"):
        if hasattr(manager,attr):
            endpoint=getattr(manager,attr)
            break

    if endpoint is None:
        endpoint=_get_endpoint_via_cli()

    if endpoint is None:
        raise RuntimeError(
            "Foundry Local endpoint'i bulunamadı. 'foundry status' çalıştırıp "
            "'Web URLs' satırındaki adresi manuel olarak kullanmayı deneyin."
        )

    if hasattr(manager,"api_key"):
        api_key=manager.api_key or "not-needed"

    try:
        subprocess.run(
            ["foundry","model","load",model_alias],
            capture_output=True,text=True,timeout=120,
            encoding="utf-8",errors="ignore"
        )
    except Exception as e:
        print(f"   [uyarı] 'foundry model load' çalıştırılamadı: {e}")

    if not endpoint.rstrip("/").endswith("/v1"):
        base_url=endpoint.rstrip("/") + "/v1"
    else:
        base_url=endpoint

    client=openai.OpenAI(base_url=base_url,api_key=api_key)

    model_id=getattr(model,"id",None) or model_alias

    input_payload=[
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text","text": prompt_text},
                {
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{base64_image}",
                    "media_type": media_type
                }
            ]
        }
    ]

    response=None
    thinking_disabled=False

    try:
        response=client.responses.create(
            model=model_id,
            input=input_payload,
            max_output_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        thinking_disabled=True
    except Exception as e:
        print(
            f"   [bilgi] enable_thinking=False desteklenmiyor gibi görünüyor "
            f"({e}), standart modla devam ediliyor..."
        )

    if response is None:
        response=client.responses.create(
            model=model_id,
            input=input_payload,
            max_output_tokens=max_tokens
        )

    text=getattr(response,"output_text",None)

    if not text:
        parts=[]
        for item in getattr(response,"output",[]):
            if getattr(item,"type",None) == "message":
                for c in getattr(item,"content",[]):
                    if getattr(c,"type",None) == "output_text":
                        parts.append(c.text)
        text="\n".join(parts).strip()

    text=(text or "").strip()
    text=re.sub(r"<think>.*?(</think>|$)","",text,flags=re.DOTALL).strip()

    status=getattr(response,"status",None)
    is_incomplete_flag=status == "incomplete"
    looks_cut_off=bool(text) and text[-1] not in ".!?\"'”)"

    if is_incomplete_flag or looks_cut_off:
        print(
            "\n   [UYARI] Cevap yarıda kesilmiş olabilir "
            "(token limiti dolmuş olabilir). max_tokens değerini "
            "artırmayı deneyebilirsin.\n"
        )

    if not text:
        raise RuntimeError(
            "Model görsel analizinden boş bir cevap döndürdü "
            f"(thinking_disabled={thinking_disabled}, status={status}). "
            "max_tokens değerini artırıp tekrar deneyin."
        )

    return text