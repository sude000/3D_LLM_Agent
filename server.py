import asyncio
import json
import os
import websockets
from google import genai
from google.genai import types

# ---------------------------------------------------------
# KONFİGÜRASYON VE GEMINI CLIENT KURULUMU
# ---------------------------------------------------------
GEMINI_API_KEY = "BURAYA_GEMINI_API_KEY_YAZIN"  # API Key'inizi buraya yapıştırın

client = genai.Client(api_key=GEMINI_API_KEY)

HOST = "localhost"
PORT = 8080

# Canlı Sahne Belleği ve Diyalog Geçmişi Belleği
CURRENT_SCENE_STATE = {"objects": []}
CHAT_HISTORY = []  # Diyalog takibi için basit bellek

def load_mock_scene():
    """Unity bağlı değilse yedek mock_scene.json dosyasını yükler."""
    global CURRENT_SCENE_STATE
    if os.path.exists("mock_scene.json"):
        with open("mock_scene.json", "r", encoding="utf-8") as f:
            CURRENT_SCENE_STATE = json.load(f)
            print("[Python Mock Engine] mock_scene.json başarıyla yüklendi.")

def build_system_instruction():
    """Sahnedeki canlı nesnelere göre dynamic system prompt oluşturur."""
    objects_summary = json.dumps(CURRENT_SCENE_STATE.get("objects", []), ensure_ascii=False)
    
    return f"""
Sen 3B bir sanal ortamda nesneleri yöneten akıllı bir yapay zeka ajanısın.
Şu anki sahnede bulunan nesneler ve mevcut konumları şunlardır:
{objects_summary}

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
  "question": "Kullanıcıya sorulacak netleştirme sorusu"
}}

3. EĞER KOMUT UYGULANAMAZ VEYA İMKANSIZSA (Sahnede olmayan nesne, fizik dışı komut vb.):
{{
  "type": "reject_command",
  "reason": "Reddetme gerekçesi"
}}

KURALLAR:
- Sahnede olmayan nesneler için ASLA "action" üretme, "reject_command" döndür.
- Hangi nesnenin kastedildiği belirsizse ("nesneyi kaldır", "onu sağa kaydır") "ask_user" üret.
- Yanıtında JSON dışında HİÇBİR açıklama veya markdown yazısı yazma.
- "pos" mutlak koordinat değil, ofset [dx, dy, dz] değeridir.
"""

def ask_gemini(user_prompt: str) -> str:
    """Kullanıcının metin komutunu, geçmiş konuşmayı ve canlı sahne durumunu Gemini API'ye gönderir."""
    global CHAT_HISTORY
    try:
        dynamic_instruction = build_system_instruction()
        
        # Diyalog geçmişini de modele aktarmak için paketliyoruz
        contents = []
        for turn in CHAT_HISTORY:
            contents.append(turn)
        contents.append(user_prompt)

        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=dynamic_instruction,
                response_mime_type="application/json"
            ),
        )
        
        # Başarılı yanıtta geçmişe ekle (Son 6 turu tut)
        CHAT_HISTORY.append(f"Kullanıcı: {user_prompt}")
        CHAT_HISTORY.append(f"Ajan Yanıtı: {response.text}")
        if len(CHAT_HISTORY) > 6:
            CHAT_HISTORY = CHAT_HISTORY[-6:]

        return response.text
    except Exception as e:
        print(f"[Python] Gemini API Hatalı Yanıt Döndü: {e}")
        return None

async def handle_client(websocket):
    global CURRENT_SCENE_STATE
    print(f"\n[Python] İstemci bağlandı: {websocket.remote_address}")
    
    try:
        while True:
            user_input = await asyncio.to_thread(input, "\n[Kullanıcı] Türkçe Komut Girin (Çıkış için 'q'): ")
            
            if user_input.strip().lower() == 'q':
                print("[Python] 'q' komutu algılandı. Sunucu kapatılıyor...")
                await websocket.close(code=1000, reason="Kullanıcı çıkış yaptı")
                asyncio.get_event_loop().stop()
                break
                
            if not user_input.strip():
                continue

            print(f"[Python] Gemini'ye soruluyor...")
            raw_response = ask_gemini(user_input)
            
            if raw_response:
                try:
                    parsed_response = json.loads(raw_response)
                    res_type = parsed_response.get("type")

                    # DURUM 1: NET HAREKET EYLEMİ
                    if res_type == "action":
                        print(f"[Python] Eylem Komutu Üretildi: {raw_response}")
                        await websocket.send(raw_response)
                        
                        # Unity'den güncel sahne durumunun gelmesini bekle
                        try:
                            unity_ack = await websocket.recv()
                            data = json.loads(unity_ack)
                            if data.get("type") == "scene_state":
                                CURRENT_SCENE_STATE = data
                                print(f"[Python] Unity Güncel Sahne Durumu Alındı. Obje Sayısı: {len(CURRENT_SCENE_STATE.get('objects', []))}")
                        except websockets.exceptions.ConnectionClosed:
                            print("[Python] Unity bağlantısı kapandı.")
                            break

                    # DURUM 2: BELİRSİZLİK (Soru Sorma)
                    elif res_type == "ask_user":
                        question = parsed_response.get("question", "Lütfen talebinizi netleştirin.")
                        print(f"\n🤖 [Ajan Soru Soruyor]: {question}")
                        # Unity'ye veri GÖNDERİLMEZ ve recv() BEKLENMEZ. Doğrudan bir sonraki input() döngüsüne geçilir.

                    # DURUM 3: UYGULANAMAZ / REDDETME
                    elif res_type == "reject_command":
                        reason = parsed_response.get("reason", "Bu komut uygulanamaz.")
                        print(f"\n🚫 [Ajan Komutu Reddetti]: {reason}")
                        # Unity'ye veri GÖNDERİLMEZ ve recv() BEKLENMEZ.

                    else:
                        print(f"[Python] Tanımsız yanıt tipi: {raw_response}")

                except json.JSONDecodeError:
                    print(f"[Python] Hata: LLM geçerli bir JSON üretmedi -> {raw_response}")
            else:
                print("[Python] Komut işlenemedi.")

    except websockets.exceptions.ConnectionClosedOK:
        print("[Python] Bağlantı temiz bir şekilde kapatıldı.")
    except Exception as e:
        print(f"[Python] Beklenmeyen hata: {e}")
    finally:
        print("[Python] Oturum sonlandırıldı.")

async def main():
    load_mock_scene()
    print(f"[Python] WebSocket Sunucusu Başlatılıyor ws://{HOST}:{PORT}...")
    server = await websockets.serve(handle_client, HOST, PORT)
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, RuntimeError):
        print("\n[Python] Sunucu başarıyla kapatıldı.")