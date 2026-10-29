import json
import os
import time
from google import genai
from google.genai import types

# ---------------------------------------------------------
# KONFİGÜRASYON VE GEMINI CLIENT KURULUMU
# ---------------------------------------------------------
GEMINI_API_KEY = "BURAYA_GEMINI_API_KEY_YAZIN"  # Google AI Studio API Key'inizi yapıştırın

client = genai.Client(api_key=GEMINI_API_KEY)

# Test sırasında kullanılacak örnek sahne durumu
MOCK_SCENE_STATE = {
    "objects": [
        {"name": "cube", "position": [0.0, 0.0, 0.0]},
        {"name": "sphere", "position": [3.0, 0.0, 0.0]},
        {"name": "cylinder", "position": [-3.0, 0.0, 0.0]}
    ]
}

def build_system_instruction():
    objects_summary = json.dumps(MOCK_SCENE_STATE.get("objects", []), ensure_ascii=False)
    
    return f"""
Sen 3B bir sanal ortamda nesneleri yöneten bir yapay zeka ajanısın.
Şu anki sahnede bulunan nesneler ve mevcut konumları şunlardır:
{objects_summary}

Kullanıcının Türkçe komutlarını analiz et ve SADECE aşağıdaki 3 JSON formatından BİRİNİ üret:

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
  "question": "Netleştirme sorusu"
}}

3. EĞER KOMUT UYGULANAMAZ VEYA İMKANSIZSA (Sahnede olmayan nesne, fizik dışı komut vb.):
{{
  "type": "reject_command",
  "reason": "Reddetme gerekçesi"
}}

KURALLAR:
- Sahnede olmayan nesneler için ASLA "action" üretme, "reject_command" döndür.
- Nesne adı belirtilmemişse ("şunu kaydır", "nesneyi kaldır") "ask_user" ile soru sor.
- Yanıtında JSON dışında HİÇBİR açıklama yazma.
"""

def test_command(user_prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=build_system_instruction(),
                response_mime_type="application/json",
                temperature=0.0  # Testlerin tutarlı (deterministik) olması için 0 yapıyoruz
            ),
        )
        return response.text
    except Exception as e:
        print(f"API Hatalı Yanıt: {e}")
        return None

def run_dataset_tests():
    dataset_path = "test_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Hata: {dataset_path} bulunamadı!")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_tests = 0
    passed_tests = 0

    all_categories = [
        ("BELİRSİZ KOMUTLAR (AMBIGUOUS)", data.get("ambiguous_commands", [])),
        ("UYGULANAMAZ KOMUTLAR (INVALID)", data.get("invalid_commands", []))
    ]

    print("=" * 70)
    print("🎯 OTO-TEST BAŞLATILIYOR: Refusal & Clarification Test Kümesi")
    print("=" * 70)

    for cat_name, commands in all_categories:
        print(f"\n--- {cat_name} ---")
        for item in commands:
            total_tests += 1
            cmd_id = item["id"]
            cmd_text = item["command"]
            expected = item["expected_type"]

            raw_res = test_command(cmd_text)
            actual_type = "UNKNOWN"
            reason_or_q = ""

            if raw_res:
                try:
                    parsed = json.loads(raw_res)
                    actual_type = parsed.get("type", "UNKNOWN")
                    if actual_type == "ask_user":
                        reason_or_q = f"Soru: {parsed.get('question')}"
                    elif actual_type == "reject_command":
                        reason_or_q = f"Gerekçe: {parsed.get('reason')}"
                except json.JSONDecodeError:
                    actual_type = "PARSE_ERROR"

            is_pass = (actual_type == expected)
            if is_pass:
                passed_tests += 1
                status_str = "✅ BAŞARILI"
            else:
                status_str = "❌ HATA"

            print(f"[{cmd_id:02d}] Komut: '{cmd_text}'")
            print(f"     Beklenen: {expected} | Gelen: {actual_type} -> {status_str}")
            if reason_or_q:
                print(f"     Ajan Yanıtı: {reason_or_q}")
            
            # API Hız Sınırına (Rate Limit) takılmamak için kısa bir bekleme
            time.sleep(1)

    print("\n" + "=" * 70)
    accuracy = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"📊 TEST SONUÇ RAPORU:")
    print(f"   Toplam Test Sayısı: {total_tests}")
    print(f"   Başarılı Test: {passed_tests}")
    print(f"   Hatalı Test: {total_tests - passed_tests}")
    print(f"   🔥 DOĞRULUK ORANI (Accuracy): %{accuracy:.1f}")
    print("=" * 70)

if __name__ == "__main__":
    run_dataset_tests()