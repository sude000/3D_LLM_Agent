import asyncio
import copy
import json
import math
import os
import threading
import websockets
import gradio as gr
from google import genai
from google.genai import types

# STT ve Türkçe Normalizasyon adaptörünü içe aktarıyoruz
from stt_adapter import transcribe_audio, normalize_turkish_text, WHISPER_AVAILABLE

# ---------------------------------------------------------
# KONFİGÜRASYON VE GEMINI CLIENT KURULUMU
# ---------------------------------------------------------
GEMINI_API_KEY = "BURAYA_API_KEY_YAZIN"  # Kendi Gemini API Key'inizi yapıştırın

client = genai.Client(api_key=GEMINI_API_KEY)

HOST = "localhost"
PORT = 8080
COLLISION_MIN_DISTANCE = 1.0  # Unity standart 1x1x1 nesneler için minimum güvenli merkez mesafesi

# Canlı Sahne Belleği, Uzamsal Durum Geçmişi, Diyalog Belleği ve WebSocket Bağlantısı
CURRENT_SCENE_STATE = {"objects": []}
INITIAL_SCENE_STATE = None
POSITION_HISTORY = {}
CHAT_HISTORY = []
ACTIVE_WEBSOCKET = None
WS_LOOP = None

# Bekleyen onay paketleri ("mock" veya "collision")
PENDING_DECISION = None


def is_unity_connected() -> bool:
    """Unity WebSocket bağlantısının aktif olup olmadığını kontrol eder."""
    return ACTIVE_WEBSOCKET is not None and WS_LOOP is not None


def get_status_markdown() -> str:
    """Üst paneldeki STT ve Unity bağlantı durum metnini dinamik üretir."""
    stt_badge = (
        "🟢 **Whisper STT Aktif**"
        if WHISPER_AVAILABLE
        else "🟡 **Whisper Devre Dışı (Yalnızca Yazılı)**"
    )
    unity_badge = (
        "🟢 **Unity Bağlı (Canlı Sahne Aktif)**"
        if is_unity_connected()
        else "🔴 **Unity Bağlı Değil (Play Modu Kapalı / Mock Beklemede)**"
    )
    return (
        f"# 🤖 3B Sanal Ortam Yapay Zekâ Ajanı\n"
        f"**Ses Modülü:** {stt_badge}  |  **Simülasyon:** {unity_badge}  |  **Port:** `ws://{HOST}:{PORT}`"
    )


def _init_spatial_memory(scene_data: dict, force: bool = False):
    """Sahne ilk yüklendiğinde başlangıç konumlarını kaydeder."""
    global INITIAL_SCENE_STATE
    objects = scene_data.get("objects", [])
    if not objects:
        return
    if INITIAL_SCENE_STATE is None or force:
        INITIAL_SCENE_STATE = copy.deepcopy(objects)
        print(f"[Uzamsal Bellek] Başlangıç sahne konumları kaydedildi ({len(objects)} nesne).")


def record_positions_before_action(actions: list):
    """Bir hareket uygulanmadan hemen önce etkilenen nesnelerin mevcut konumunu geçmişe kaydeder."""
    current_map = {
        str(obj.get("name", "")).lower(): list(obj.get("position", [0.0, 0.0, 0.0]))
        for obj in CURRENT_SCENE_STATE.get("objects", [])
    }
    for act in actions:
        if act.get("action") == "move":
            target = str(act.get("target", "")).lower()
            if target in current_map:
                POSITION_HISTORY.setdefault(target, []).append(current_map[target])


def build_spatial_history_summary() -> str:
    """Gemini'nin başlangıç ve önceki konumları doğrudan görebilmesi için bellek özeti üretir."""
    init_list = INITIAL_SCENE_STATE or []
    prev_map = {
        name: stack[-1]
        for name, stack in POSITION_HISTORY.items()
        if stack
    }
    return (
        f"Başlangıç Konumları (Initial Positions): {json.dumps(init_list, ensure_ascii=False)}\n"
        f"Nesnelerin Bir Önceki Konumları (Previous Positions): {json.dumps(prev_map, ensure_ascii=False)}"
    )


def load_mock_scene():
    """Unity henüz bağlı değilse yedek mock_scene.json dosyasını yükler."""
    global CURRENT_SCENE_STATE
    if os.path.exists("mock_scene.json"):
        with open("mock_scene.json", "r", encoding="utf-8") as f:
            CURRENT_SCENE_STATE = json.load(f)
            _init_spatial_memory(CURRENT_SCENE_STATE, force=True)
            print("[Mock Engine] mock_scene.json yüklendi.")


def check_action_collisions(actions: list) -> dict:
    """
    Üretilen hareketlerin sahnedeki diğer nesnelerle çarpışıp çarpışmayacağını deterministik olarak hesaplar.
    Dönüş:
      {
        "has_collision": bool,
        "collisions": [(hareketli_nesne, carpisilan_nesne, hedef_pos), ...],
        "safe_actions": [...],
        "unsafe_actions": [...]
      }
    """
    scene_objs = CURRENT_SCENE_STATE.get("objects", [])
    obj_info = {
        str(o.get("name", "")).lower(): {
            "pos": list(o.get("position", [0.0, 0.0, 0.0])),
            "color": str(o.get("color", "")),
        }
        for o in scene_objs
    }

    # Tüm nesnelerin hareket sonrası nihai konum haritasını çıkar
    final_positions = {name: list(info["pos"]) for name, info in obj_info.items()}
    action_targets = {}
    for act in actions:
        if act.get("action") == "move":
            t = str(act.get("target", "")).lower()
            p = act.get("pos", [0.0, 0.0, 0.0])
            if t in final_positions and len(p) == 3:
                cur = final_positions[t]
                final_positions[t] = [
                    round(cur[0] + p[0], 3),
                    round(cur[1] + p[1], 3),
                    round(cur[2] + p[2], 3),
                ]
                action_targets[t] = act

    collisions = []
    unsafe_targets = set()

    for t, act in action_targets.items():
        t_pos = final_positions[t]
        for other_name, other_pos in final_positions.items():
            if other_name == t:
                continue
            dist = math.sqrt(
                (t_pos[0] - other_pos[0]) ** 2
                + (t_pos[1] - other_pos[1]) ** 2
                + (t_pos[2] - other_pos[2]) ** 2
            )
            if dist < COLLISION_MIN_DISTANCE:
                collisions.append((t, other_name, t_pos))
                unsafe_targets.add(t)

    safe_actions = [a for a in actions if str(a.get("target", "")).lower() not in unsafe_targets]
    unsafe_actions = [a for a in actions if str(a.get("target", "")).lower() in unsafe_targets]

    return {
        "has_collision": len(collisions) > 0,
        "collisions": collisions,
        "safe_actions": safe_actions,
        "unsafe_actions": unsafe_actions,
        "obj_info": obj_info,
    }


def _format_obj_label(name: str, obj_info: dict) -> str:
    """Nesne adını Türkçe rengiyle birlikte okunaklı yazar (örn. 'kırmızı küp (cube)')."""
    tr_names = {
        "cube": "küp",
        "cube_2": "küp",
        "sphere": "küre",
        "cylinder": "silindir",
    }
    info = obj_info.get(name, {})
    color = info.get("color", "").strip()
    base = tr_names.get(name, name)
    return f"{color} {base} (`{name}`)" if color else f"`{name}`"


def build_system_instruction() -> str:
    """Sahnedeki canlı nesnelere, başlangıç konumlarına ve geçmiş konumlara göre sistem yönergesi oluşturur."""
    objects_summary = json.dumps(CURRENT_SCENE_STATE.get("objects", []), ensure_ascii=False)
    spatial_history = build_spatial_history_summary()

    return f"""
Sen 3B bir sanal ortamda nesneleri yöneten akıllı bir yapay zeka ajanısın.
Şu anki sahnede bulunan nesneler, renkleri ve GÜNCEL konumları şunlardır:
{objects_summary}

Uzamsal Konum Geçmişi (Bellek):
{spatial_history}

Kullanıcının Türkçe doğal dil komutlarını analiz et ve SADECE aşağıdaki 3 JSON formatından BİRİNİ üret:

1. EĞER KOMUT NET VE UYGULANABİLİMSE:
{{
  "type": "action",
  "actions": [
    {{
      "action": "move",
      "target": "nesne_adi",
      "pos": [x_offset, y_offset, z_offset]
    }}
  ]
}}

2. EĞER KOMUT EKSİK VEYA BELİRSİZSE (Hangi nesne olduğu net değilse veya hedef belirsizse):
{{
  "type": "ask_user",
  "question": "Kullanıcıya sorulacak netleştirme sorusu (Türkçe)"
}}

3. EĞER KOMUT UYGULANAMAZ VEYA İMKANSIZSA (Sahnede olmayan nesne, fizik dışı komut vb.):
{{
  "type": "reject_command",
  "reason": "Reddetme gerekçesi (Türkçe)"
}}

KURALLAR:
- Kullanıcı bir nesneyi rengiyle tanımlarsa ("mavi küp", "kırmızı olan"), sahnedeki "color" alanına bakarak doğru "target" nesnesini eşleştir.
- Sahnede aynı türden birden fazla nesne varsa ve kullanıcı renk/isim belirtmeden genel konuştuysa ("küpü kaydır"), renklerini belirterek hangi küpü istediğini sor.
- Eğer kullanıcı komutta hiçbir nesne adı veya rengi belirtmemişse ("5 birim yukarı kaldır", "2 birim sağa kaydır"), önceki tamamlanmış veya iptal edilmiş komutlardaki nesneyi varsayma; mutlaka "ask_user" ile hangi nesneyi hareket ettirmek istediğini sor. (İstisna: "tüm nesneleri başlangıç konumuna döndür" veya "sahneyi sıfırla" gibi tüm sahneyi kapsayan komutlar).
- BAŞLANGIÇ KONUMUNA DÖNME KURALI: Kullanıcı bir nesneyi veya tüm nesneleri "başlangıç konumuna döndür / ilk yerine getir / sıfırla" derse, yukarıdaki "Başlangıç Konumları" listesindeki hedef koordinat [x0, y0, z0] ile nesnenin GÜNCEL konumu [x, y, z] arasındaki farkı hesapla: pos = [x0 - x, y0 - y, z0 - z]. Eğer nesne zaten başlangıç konumundaysa bunu belirterek "reject_command" döndür.
- ÖNCEKİ KONUMA DÖNME KURALI: Kullanıcı bir nesneyi "önceki konumuna taşı / geri al / eski yerine koy" derse, "Nesnelerin Bir Önceki Konumları" sözlüğüne bak. İlgili nesnenin kaydı varsa pos = [x_prev - x, y_prev - y, z_prev - z] vektörünü üret. Kaydı yoksa (henüz hareket ettirilmemişse) bunu belirterek "reject_command" döndür.
- Sahnede olmayan nesneler için ASLA "action" üretme, "reject_command" döndür.
- "pos" mutlak koordinat değil, ofset [dx, dy, dz] değeridir ("sağa" +x, "sola" -x, "yukarı/havaya" +y, "aşağı" -y, "ileri" +z, "geri" -z).
- Miktar belirtilmemişse varsayılan adımı 2 birim al.
- Yanıtında JSON dışında HİÇBİR açıklama veya markdown yazısı yazma.
"""


def ask_gemini(user_prompt: str) -> str:
    """Kullanıcının normalize edilmiş komutunu, geçmiş konuşmayı ve sahne durumunu Gemini'ye sorar."""
    global CHAT_HISTORY
    try:
        dynamic_instruction = build_system_instruction()

        contents = []
        for turn in CHAT_HISTORY:
            contents.append(turn)
        contents.append(user_prompt)

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=dynamic_instruction,
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        CHAT_HISTORY.append(f"Kullanıcı: {user_prompt}")
        CHAT_HISTORY.append(f"Ajan Yanıtı: {response.text}")
        if len(CHAT_HISTORY) > 8:
            CHAT_HISTORY = CHAT_HISTORY[-8:]

        return response.text
    except Exception as e:
        print(f"[Gemini Hata] API yanıt vermedi: {e}")
        return None


# ---------------------------------------------------------
# ARKA PLAN WEBSOCKET SUNUCUSU (UNITY BAĞLANTISI)
# ---------------------------------------------------------
async def ws_handler(websocket):
    global ACTIVE_WEBSOCKET, CURRENT_SCENE_STATE, POSITION_HISTORY
    ACTIVE_WEBSOCKET = websocket
    first_state_received = False
    print(f"[WebSocket] Unity istemcisi bağlandı: {websocket.remote_address}")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if isinstance(data, dict) and data.get("type") == "scene_state":
                    CURRENT_SCENE_STATE = data
                    if not first_state_received:
                        _init_spatial_memory(data, force=True)
                        POSITION_HISTORY = {}
                        first_state_received = True
                    print(f"[WebSocket] Canlı sahne durumu güncellendi: {len(data.get('objects', []))} nesne.")
                else:
                    print(f"[WebSocket] Unity geri bildirimi: {message}")
            except json.JSONDecodeError:
                print(f"[WebSocket] Unity mesajı: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("[WebSocket] Unity bağlantısı kesildi.")
    finally:
        if ACTIVE_WEBSOCKET == websocket:
            ACTIVE_WEBSOCKET = None


def start_ws_server_in_thread():
    """WebSocket sunucusunu Gradio'yu bloklamadan ayrı bir arka plan thread'inde çalıştırır."""
    global WS_LOOP
    WS_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(WS_LOOP)

    async def run_server():
        print(f"[WebSocket] Sunucu başlatılıyor: ws://{HOST}:{PORT}")
        async with websockets.serve(ws_handler, HOST, PORT):
            await asyncio.Future()

    WS_LOOP.run_until_complete(run_server())


def send_actions_to_unity(actions: list) -> tuple[bool, str]:
    """Seçilen eylem listesini JSON paketine çevirip WebSocket üzerinden Unity'ye iletir."""
    if not is_unity_connected():
        return False, "Unity bağlı değil."

    payload = json.dumps({"type": "action", "actions": actions}, ensure_ascii=False)
    try:
        future = asyncio.run_coroutine_threadsafe(ACTIVE_WEBSOCKET.send(payload), WS_LOOP)
        future.result(timeout=3.0)
        return True, "✅ Komut Unity sahnesine iletildi."
    except Exception as e:
        return False, f"⚠️ Unity'ye gönderim hatası: {e}"


def apply_actions_to_mock_scene(actions: list):
    """Onay verildiğinde mock_scene belleğindeki nesne koordinatlarını günceller."""
    for act in actions:
        if act.get("action") == "move":
            target_name = str(act.get("target", "")).lower()
            pos = act.get("pos", [0.0, 0.0, 0.0])
            for obj in CURRENT_SCENE_STATE.get("objects", []):
                if str(obj.get("name", "")).lower() == target_name and len(pos) == 3:
                    cur_pos = obj.get("position", [0.0, 0.0, 0.0])
                    obj["position"] = [
                        round(cur_pos[0] + pos[0], 2),
                        round(cur_pos[1] + pos[1], 2),
                        round(cur_pos[2] + pos[2], 2),
                    ]


def dispatch_actions(actions: list, source_label: str, force_mock: bool = False) -> str:
    """Eylemleri geçmişe kaydeder ve Unity'ye veya onaylı mock sahnesine uygular."""
    record_positions_before_action(actions)
    if is_unity_connected() and not force_mock:
        _, status_text = send_actions_to_unity(actions)
    else:
        apply_actions_to_mock_scene(actions)
        status_text = "✅ Onay verildi: Komut `mock_scene` simülasyonu üzerinde işlendi."

    action_lines = [
        f"• **{act.get('target')}** nesnesi `{act.get('pos')}` vektörü kadar hareket ettirildi."
        for act in actions
    ]
    return (
        f"**[Eylem Onaylandı]** ({source_label})\n"
        + "\n".join(action_lines)
        + f"\n\n*{status_text}*"
    )


# ---------------------------------------------------------
# ORTAK KOMUT İŞLEME HATTI (ÇARPIŞMA KALKANI + MOCK ONAY)
# ---------------------------------------------------------
def execute_agent_pipeline(normalized_command: str, raw_source_label: str) -> tuple[str, bool, str, str]:
    """
    Dönüş: (sohbet_mesaji, onay_bekleniyor_mu, evet_buton_etiketi, hayir_buton_etiketi)
    """
    global PENDING_DECISION
    raw_response = ask_gemini(normalized_command)
    if not raw_response:
        return "❌ Yapay zekâ servisine şu an ulaşılamıyor. Lütfen tekrar deneyin.", False, "✅ Evet, devam et", "❌ Hayır, iptal et"

    try:
        parsed = json.loads(raw_response)
        res_type = parsed.get("type")

        if res_type == "action":
            actions = parsed.get("actions", [])

            # 1. ÖNCE ÇARPIŞMA / ÜST ÜSTE BİNME KONTROLÜ YAP
            col_report = check_action_collisions(actions)
            if col_report["has_collision"]:
                obj_info = col_report["obj_info"]
                safe_actions = col_report["safe_actions"]

                conflict_lines = [
                    f"• {_format_obj_label(src, obj_info)} ile {_format_obj_label(dst, obj_info)} `{pos}` noktasında çarpışıyor."
                    for src, dst, pos in col_report["collisions"]
                ]

                if safe_actions:
                    safe_labels = ", ".join(
                        _format_obj_label(str(a.get("target", "")).lower(), obj_info)
                        for a in safe_actions
                    )
                    yes_label = f"✅ Yalnızca güvenli olanı uygula ({', '.join(a.get('target') for a in safe_actions)})"
                    question_line = f"Yalnızca {safe_labels} nesnesini hareket ettirelim mi, yoksa işlemi iptal mi edelim?"
                else:
                    yes_label = "⚠️ Çarpışmaya rağmen yine de uygula"
                    question_line = "Çarpışma riskine rağmen hareketi yine de uygulayalım mı, yoksa iptal mi edelim?"

                PENDING_DECISION = {
                    "kind": "collision",
                    "all_actions": actions,
                    "safe_actions": safe_actions,
                    "source_label": raw_source_label,
                }
                warn_msg = (
                    "🚨 **Çarpışma Riski Tespit Edildi!**\n"
                    + "\n".join(conflict_lines)
                    + f"\n\n**{question_line}**\n"
                    "*(Seçim yapılana kadar yeni komut girişi kilitlenmiştir.)*"
                )
                return warn_msg, True, yes_label, "❌ İşlemi iptal et"

            # 2. ÇARPIŞMA YOKSA VE UNITY BAĞLIYSA DOĞRUDAN UYGULA
            if is_unity_connected():
                reply = dispatch_actions(actions, raw_source_label)
                return reply, False, "✅ Evet, devam et", "❌ Hayır, iptal et"

            # 3. ÇARPIŞMA YOK AMA UNITY BAĞLI DEĞİLSE MOCK ONAYI İSTE
            PENDING_DECISION = {
                "kind": "mock",
                "all_actions": actions,
                "safe_actions": actions,
                "source_label": raw_source_label,
            }
            preview_lines = [
                f"• **{act.get('target')}** → `{act.get('pos')}`"
                for act in actions
            ]
            prompt_msg = (
                "⚠️ **Unity Bağlı Değil!**\n"
                "Unity editöründe Play modu aktif görünmüyor. Aşağıdaki eylem planı hazırlandı:\n"
                + "\n".join(preview_lines)
                + "\n\n**Unity aktif değil, `mock_scene` üzerinden işleme devam edilsin mi?**\n"
                "*(Devam etmek için aşağıdaki seçeneklerden birini seçin; seçim yapılana kadar yeni komut girişi kilitlenmiştir.)*"
            )
            return prompt_msg, True, "✅ Evet, devam et", "❌ Hayır, iptal et"

        elif res_type == "ask_user":
            question = parsed.get("question", "Lütfen hangi nesneyi kastettiğinizi netleştirin.")
            return f"🤔 **Netleştirme Sorusu:** {question}", False, "✅ Evet, devam et", "❌ Hayır, iptal et"

        elif res_type == "reject_command":
            reason = parsed.get("reason", "Bu komut mevcut sahne koşullarında uygulanamaz.")
            return f"🚫 **Komut Reddedildi:** {reason}", False, "✅ Evet, devam et", "❌ Hayır, iptal et"

        else:
            return f"⚠️ Beklenmeyen ajan yanıtı: `{raw_response}`", False, "✅ Evet, devam et", "❌ Hayır, iptal et"

    except json.JSONDecodeError:
        return f"⚠️ Ajan geçersiz JSON üretti: `{raw_response}`", False, "✅ Evet, devam et", "❌ Hayır, iptal et"


def _ui_lock_updates(lock: bool, yes_label: str = "✅ Evet, devam et", no_label: str = "❌ Hayır, iptal et"):
    """Onay beklenirken girdi bileşenlerini kilitler ve buton etiketlerini günceller."""
    return (
        gr.update(visible=lock),               # confirm_row
        gr.update(value=yes_label),            # confirm_yes_btn
        gr.update(value=no_label),             # confirm_no_btn
        gr.update(interactive=not lock),       # audio_input
        gr.update(interactive=not lock),       # send_audio_btn
        gr.update(interactive=not lock),       # text_input
        gr.update(interactive=not lock),       # send_text_btn
    )


# ---------------------------------------------------------
# GRADIO ARAYÜZ OLAY FONKSİYONLARI
# ---------------------------------------------------------
def handle_text_submit(user_text: str, history: list):
    """Yazılı komut gönderildiğinde çalışır."""
    if PENDING_DECISION is not None:
        return ("", history, *_ui_lock_updates(True))

    if not user_text or not user_text.strip():
        return ("", history, *_ui_lock_updates(False))

    normalized = normalize_turkish_text(user_text)
    if not normalized:
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": "⚠️ Geçerli bir komut algılanamadı."})
        return ("", history, *_ui_lock_updates(False))

    display_user = f"⌨️ {user_text}\n*(Normalize: `{normalized}`)*"
    agent_reply, needs_confirm, yes_lbl, no_lbl = execute_agent_pipeline(normalized, "Yazılı Girdi")

    history.append({"role": "user", "content": display_user})
    history.append({"role": "assistant", "content": agent_reply})
    return ("", history, *_ui_lock_updates(needs_confirm, yes_lbl, no_lbl))


def handle_audio_submit(audio_path: str, history: list):
    """Mikrofon kaydı gönderildiğinde çalışır."""
    if PENDING_DECISION is not None:
        return (None, history, *_ui_lock_updates(True))

    if not audio_path:
        history.append({"role": "assistant", "content": "⚠️ Ses kaydı bulunamadı. Mikrofona konuşup tekrar gönderin."})
        return (None, history, *_ui_lock_updates(False))

    success, result_text = transcribe_audio(audio_path)

    if not success:
        history.append({"role": "assistant", "content": f"🎙️ **Ses Modülü Bilgisi:** {result_text}"})
        return (None, history, *_ui_lock_updates(False))

    display_user = f"🎙️ *(Sesli Komut - Whisper)*: **`{result_text}`**"
    agent_reply, needs_confirm, yes_lbl, no_lbl = execute_agent_pipeline(result_text, "Sesli Girdi")

    history.append({"role": "user", "content": display_user})
    history.append({"role": "assistant", "content": agent_reply})
    return (None, history, *_ui_lock_updates(needs_confirm, yes_lbl, no_lbl))


def handle_decision_confirm(history: list):
    """Onay butonuna basıldığında (çarpışma güvenli filtresi veya mock onayı) eylemi uygular."""
    global PENDING_DECISION
    if PENDING_DECISION is None:
        return (history, *_ui_lock_updates(False))

    pending = PENDING_DECISION
    PENDING_DECISION = None

    kind = pending["kind"]
    source_label = pending["source_label"]

    if kind == "collision":
        # Güvenli eylemler varsa sadece onları uygula; hiç güvenli yoksa kullanıcının zorunlu onayını uygula
        chosen_actions = pending["safe_actions"] if pending["safe_actions"] else pending["all_actions"]
        if not is_unity_connected():
            # Unity kapalıysa mock onayı adımına geçir
            PENDING_DECISION = {
                "kind": "mock",
                "all_actions": chosen_actions,
                "safe_actions": chosen_actions,
                "source_label": source_label,
            }
            preview_lines = [f"• **{act.get('target')}** → `{act.get('pos')}`" for act in chosen_actions]
            history.append({
                "role": "assistant",
                "content": (
                    "⚠️ **Unity Bağlı Değil!**\n"
                    "Seçilen güvenli eylem planı hazır:\n"
                    + "\n".join(preview_lines)
                    + "\n\n**`mock_scene` üzerinde uygulansın mı?**"
                ),
            })
            return (history, *_ui_lock_updates(True, "✅ Evet, devam et", "❌ Hayır, iptal et"))

        reply = dispatch_actions(chosen_actions, source_label)
        history.append({"role": "assistant", "content": reply})
        return (history, *_ui_lock_updates(False))

    # kind == "mock"
    reply = dispatch_actions(pending["all_actions"], f"{source_label} - Mock")
    history.append({"role": "assistant", "content": reply})
    return (history, *_ui_lock_updates(False))


def handle_decision_cancel(history: list):
    """İptal butonuna basıldığında bekleyen eylemi iptal eder, hayalet bağlamı temizler ve kilidi açar."""
    global PENDING_DECISION, CHAT_HISTORY
    pending = PENDING_DECISION
    PENDING_DECISION = None

    if len(CHAT_HISTORY) >= 2:
        CHAT_HISTORY = CHAT_HISTORY[:-2]
    CHAT_HISTORY.append("Sistem Notu: Önceki eylem kullanıcı tarafından iptal edildi; yeni komutta nesne belirtilmezse mutlaka sor.")

    cancel_reason = (
        "Çarpışma riski nedeniyle işlem iptal edildi; hiçbir nesne hareket ettirilmedi."
        if pending and pending.get("kind") == "collision"
        else "`mock_scene` üzerinde değişiklik yapılmadı."
    )
    history.append({
        "role": "assistant",
        "content": f"🛑 **İşlem İptal Edildi:** {cancel_reason}"
    })
    return (history, *_ui_lock_updates(False))


def reset_dialogue_memory():
    """Sohbet geçmişini, diyalog belleğini ve bekleyen onay kilidini sıfırlar."""
    global CHAT_HISTORY, PENDING_DECISION
    CHAT_HISTORY = []
    PENDING_DECISION = None
    return ([], None, "", *_ui_lock_updates(False))


# ---------------------------------------------------------
# GRADIO ARAYÜZ TASARIMI
# ---------------------------------------------------------
def create_ui():
    with gr.Blocks(title="3D LLM Ajanı - Ses & Sohbet Kontrol Paneli") as demo:
        status_banner = gr.Markdown(value=get_status_markdown())

        status_timer = gr.Timer(value=2.0)
        status_timer.tick(fn=get_status_markdown, inputs=[], outputs=[status_banner])

        chatbot = gr.Chatbot(
            label="Ajan Etkileşim Ekranı",
            height=400,
        )

        with gr.Row(visible=False) as confirm_row:
            confirm_yes_btn = gr.Button("✅ Evet, devam et", variant="primary")
            confirm_no_btn = gr.Button("❌ Hayır, iptal et", variant="stop")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎙️ Sesli Komut Kaydet",
                )
                send_audio_btn = gr.Button("🎤 Sesi Çözümle ve Gönder", variant="primary")

            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="⌨️ Yazılı Komut (Yedek / Doğrudan Girdi)",
                    placeholder="Örn: Mavi küpü 3 birim yukarı kaldır...",
                    lines=2,
                )
                with gr.Row():
                    send_text_btn = gr.Button("📤 Yazılı Gönder", variant="secondary")
                    clear_btn = gr.Button("🗑️ Belleği Temizle")

        lockable_outputs = [
            confirm_row,
            confirm_yes_btn,
            confirm_no_btn,
            audio_input,
            send_audio_btn,
            text_input,
            send_text_btn,
        ]

        send_text_btn.click(
            fn=handle_text_submit,
            inputs=[text_input, chatbot],
            outputs=[text_input, chatbot, *lockable_outputs],
        )
        text_input.submit(
            fn=handle_text_submit,
            inputs=[text_input, chatbot],
            outputs=[text_input, chatbot, *lockable_outputs],
        )
        send_audio_btn.click(
            fn=handle_audio_submit,
            inputs=[audio_input, chatbot],
            outputs=[audio_input, chatbot, *lockable_outputs],
        )
        confirm_yes_btn.click(
            fn=handle_decision_confirm,
            inputs=[chatbot],
            outputs=[chatbot, *lockable_outputs],
        )
        confirm_no_btn.click(
            fn=handle_decision_cancel,
            inputs=[chatbot],
            outputs=[chatbot, *lockable_outputs],
        )
        clear_btn.click(
            fn=reset_dialogue_memory,
            inputs=[],
            outputs=[chatbot, audio_input, text_input, *lockable_outputs],
        )

    return demo


if __name__ == "__main__":
    load_mock_scene()

    ws_thread = threading.Thread(target=start_ws_server_in_thread, daemon=True)
    ws_thread.start()

    ui = create_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)