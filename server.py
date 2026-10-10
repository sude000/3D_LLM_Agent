import asyncio
import json
import websockets

# Sunucu konfigürasyonu
HOST = "localhost"
PORT = 8080

async def handle_client(websocket):
    print(f"[Python] Unity istemcisi bağlandı: {websocket.remote_address}")
    
    try:
        # Test mesajımızı oluşturuyoruz
        test_command = {
            "action": "move",
            "target": "cube",
            "pos": [1, 0, 0]
        }
        
        # JSON'ı string formatına çevirip Unity'ye gönderiyoruz
        message_str = json.dumps(test_command)
        print(f"[Python] Unity'ye komut gönderiliyor: {message_str}")
        await websocket.send(message_str)

        # Unity'den gelebilecek onay/yanıt mesajlarını dinliyoruz
        async for message in websocket:
            print(f"[Python] Unity'den yanıt alındı: {message}")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"[Python] Unity bağlantısı kesildi: {e}")
    except Exception as e:
        print(f"[Python] Hata oluştu: {e}")
    finally:
        print("[Python] İstemci oturumu sonlandırıldı.")

async def main():
    print(f"[Python] WebSocket Sunucusu başlatılıyor ws://{HOST}:{PORT}...")
    async with websockets.serve(handle_client, HOST, PORT):
        await asyncio.Future()  # Sunucuyu sürekli açık tutar

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Python] Sunucu kapatıldı.")