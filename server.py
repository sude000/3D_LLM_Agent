import asyncio
import json
import websockets
from google import genai
from google.genai import types

# ---------------------------------------------------------
# KONFİGÜRASYON VE GEMINI CLIENT KURULUMU
# ---------------------------------------------------------
# API Key'i kodun içinden kaldırıyoruz
GEMINI_API_KEY = "BURAYA_GEMINI_API_KEY_YAZIN"

client = genai.Client(api_key=GEMINI_API_KEY)

HOST = "localhost"
PORT = 8080

SYSTEM_INSTRUCTION = """
Sen 3B bir sanal ortamda nesneleri yöneten bir yapay zeka ajanısın.
Kullanıcının vereceği Türkçe doğal dil komutlarını ANCAK VE SADECE aşağıdaki JSON formatına dönüştüreceksin.

Beklenen JSON Formatı:
{
  "action": "move",
  "target": "cube",
  "pos": [x, y, z]
}

Kurallar:
1. Yanıtında JSON dışında HİÇBİR açıklama, selamlama veya markdown yazısı yazma.
2. "sağa" derse x pozisyonunu artır, "sola" derse x pozisyonunu azalt.
3. "yukarı/havaya" derse y pozisyonunu artır, "aşağı" derse y pozisyonunu azalt.
4. "ileri" derse z pozisyonunu artır, "geri" derse z pozisyonunu azalt.
5. Kullanıcı spesifik bir miktar belirtmediyse varsayılan adımı 2 birim al (örn: [2, 0, 0]).
"""

def ask_gemini(user_prompt: str) -> str:
    """Kullanıcının metin komutunu Gemini API'ye gönderir ve JSON string alır."""
    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json"
            ),
        )
        return response.text
    except Exception as e:
        print(f"[Python] Gemini API Hatalı Yanıt Döndü: {e}")
        return None

async def handle_client(websocket):
    print(f"\n[Python] Unity istemcisi bağlandı: {websocket.remote_address}")
    
    try:
        while True:
            # 1. Kullanıcıdan komutu alıyoruz
            user_input = await asyncio.to_thread(input, "\n[Kullanıcı] Türkçe Komut Girin (Çıkış için 'q'): ")
            
            if user_input.strip().lower() == 'q':
                print("[Python] 'q' komutu algılandı. Oturum ve sunucu kapatılıyor...")
                await websocket.close(code=1000, reason="Kullanıcı çıkış yaptı")
                asyncio.get_event_loop().stop()
                break
                
            if not user_input.strip():
                continue

            print(f"[Python] Gemini'ye soruluyor: '{user_input}'...")
            json_response = ask_gemini(user_input)
            
            if json_response:
                print(f"[Python] Gemini'den Gelen JSON Paket: {json_response}")
                # 2. JSON'ı Unity'ye yolluyoruz
                await websocket.send(json_response)
                
                # 3. Unity'den onay mesajı gelene kadar BEKLİYORUZ (Yeni input açmıyoruz)
                try:
                    unity_ack = await websocket.recv()
                    print(f"[Python] Unity Geri Bildirimi: {unity_ack}")
                except websockets.exceptions.ConnectionClosed:
                    print("[Python] Unity bağlantısı kapandı.")
                    break
            else:
                print("[Python] Komut işlenemedi, Unity'ye veri gönderilmedi.")

    except websockets.exceptions.ConnectionClosedOK:
        print("[Python] Unity bağlantısı temiz bir şekilde kapatıldı.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"[Python] Unity bağlantısı aniden kesildi: {e}")
    except Exception as e:
        print(f"[Python] Beklenmeyen hata: {e}")
    finally:
        print("[Python] İstemci oturumu sonlandırıldı.")

async def main():
    print(f"[Python] WebSocket Sunucusu Başlatılıyor ws://{HOST}:{PORT}...")
    server = await websockets.serve(handle_client, HOST, PORT)
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, RuntimeError):
        print("\n[Python] Sunucu başarıyla kapatıldı.")