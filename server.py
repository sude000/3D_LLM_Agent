import asyncio
import json
import os
import websockets
from google import genai
from google.genai import types

# ---------------------------------------------------------
# KONFİGÜRASYON VE GEMINI CLIENT KURULUMU
# ---------------------------------------------------------
GEMINI_API_KEY = "BURAYA_GEMINI_API_KEY_YAZIN"  # Google AI Studio API Key'inizi yazın

client = genai.Client(api_key=GEMINI_API_KEY)

HOST = "localhost"
PORT = 8080

# Sahne Durumu Belleği (Varsayılan olarak Mock dosyadan okur)
CURRENT_SCENE_STATE = {"objects": []}

def load_mock_scene():
    """Unity bağlı değilse yedek mock_scene.json dosyasını yükler."""
    global CURRENT_SCENE_STATE
    if os.path.exists("mock_scene.json"):
        with open("mock_scene.json", "r", encoding="utf-8") as f:
            CURRENT_SCENE_STATE = json.load(f)
            print("[Python Mock Engine] mock_scene.json başarıyla yüklendi.")

def build_system_instruction():
    """Sahnedeki canlı nesnelere göre dynamic prompt oluşturur."""
    objects_summary = json.dumps(CURRENT_SCENE_STATE.get("objects", []), ensure_ascii=False)
    
    return f"""
Sen 3B bir sanal ortamda nesneleri yöneten bir yapay zeka ajanısın.
Şu anki sahnede bulunan nesneler ve mevcut konumları şunlardır:
{objects_summary}

Kullanıcının vereceği Türkçe doğal dil komutlarını ANCAK VE SADECE aşağıdaki JSON formatına dönüştüreceksin.

Beklenen JSON Formatı:
{{
  "actions": [
    {{
      "action": "move",
      "target": "nesne_adi",
      "pos": [x_offset, y_offset, z_offset]
    }}
  ]
}}

KURALLAR VE ÖNEMLİ MATEMATİKSEL TALİMATLAR:
1. "actions" alanı her zaman bir liste (array) olmalıdır. Kullanıcı tek bir hareket istese bile 1 elemanlı bir liste üret.
2. Kullanıcı aynı cümlede birden fazla nesne hareketi isterse (örn: "Küreyi sağa, silindiri yukarı kaydır"), hepsini "actions" listesinin içine sırayla ekle.
3. "pos" alanı mutlak bir koordinat DEĞİLDİR. Mevcut konuma eklenecek OFSET/KAYDIRMA vektörüdür [dx, dy, dz].
4. ASLA nesnenin mevcut konumunu hesaplayıp yeni mutlak koordinatı yazma. Sadece hareket miktarını yaz.
5. Yön Kuralları:
   - "sağa" -> [+x, 0, 0] (Örn: 5 birim sağa -> [5, 0, 0])
   - "sola" -> [-x, 0, 0] (Örn: 5 birim sola -> [-5, 0, 0])
   - "yukarı / havaya" -> [0, +y, 0]
   - "aşağı" -> [0, -y, 0]
   - "ileri" -> [0, 0, +z] (Örn: 5 birim ileri -> [0, 0, 5])
   - "geri" -> [0, 0, -z]
6. "target" alanı sahnede bulunan nesnelerden biri olmalıdır (örn: "cube", "sphere", "cylinder").
7. Kullanıcı miktar belirtmediyse varsayılan adımı 2 birim al.
8. Yanıtında JSON dışında HİÇBİR açıklama veya markdown yazısı yazma.
"""

def ask_gemini(user_prompt: str) -> str:
    """Kullanıcının metin komutunu ve canlı sahne durumunu Gemini API'ye gönderir."""
    try:
        dynamic_instruction = build_system_instruction()
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=dynamic_instruction,
                response_mime_type="application/json"
            ),
        )
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

            print(f"[Python] Gemini'ye soruluyor (Canlı Sahne Durumu İle)...")
            json_response = ask_gemini(user_input)
            
            if json_response:
                print(f"[Python] Gemini'den Gelen JSON Paket: {json_response}")
                await websocket.send(json_response)
                
                # Unity'den güncel sahne durumunun gelmesini bekleme
                try:
                    unity_ack = await websocket.recv()
                    data = json.loads(unity_ack)
                    if data.get("type") == "scene_state":
                        CURRENT_SCENE_STATE = data
                        print(f"[Python] Unity Güncel Sahne Durumu Alındı. Obje Sayısı: {len(CURRENT_SCENE_STATE.get('objects', []))}")
                    else:
                        print(f"[Python] Unity Geri Bildirimi: {unity_ack}")
                except websockets.exceptions.ConnectionClosed:
                    print("[Python] Unity bağlantısı kapandı.")
                    break
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