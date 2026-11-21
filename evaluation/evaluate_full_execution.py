import json
import os
import time
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. KONFİGÜRASYON
# ---------------------------------------------------------
GEMINI_API_KEY = "BURAYA_API_KEY_YAZIN"  # <-- Kendi API anahtarını yapıştır
MODEL_NAME = "gemini-3.8-flash"

client = genai.Client(api_key=GEMINI_API_KEY)

# Gerçek server.py ile birebir aynı System Prompt
def build_production_system_instruction(scene_state: dict) -> str:
    objects_summary = json.dumps(scene_state.get("objects", []), ensure_ascii=False)
    return f"""
Sen 3B bir sanal ortamda nesneleri yöneten akıllı bir yapay zeka ajanısın.
Şu anki sahnede bulunan nesneler, renkleri ve mevcut konumları şunlardır:
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

2. EĞER KOMUT EKSİK VEYA BELİRSİZSE (Hangi nesne olduğu net değilse veya yön belirsizse):
{{
  "type": "ask_user",
  "question": "Kullanıcıya sorulacak netleştirme sorusu (Türkçe)"
}}

3. EĞER KOMUT UYGULANAMAZ VEYA İMKANSIZSA (Sahnede olmayan nesne, fizik dışı/çarpışma/desteklenmeyen eylem vb.):
{{
  "type": "reject_command",
  "reason": "Reddetme gerekçesi (Türkçe)"
}}

KURALLAR VE MATEMATİKSEL TALİMATLAR:
- Kullanıcı bir nesneyi rengiyle veya koordinatıyla tanımlarsa sahnedeki doğru "target" nesnesini eşleştir.
- Sahnede aynı türden birden fazla nesne varsa (örn. 2 küp) ve hangisi olduğu net değilse "ask_user" ile soru sor.
- "pos" mutlak koordinat değil, mevcut konuma eklenecek OFSET [dx, dy, dz] vektörüdür.
- Yön Kuralları: sağa=[+x,0,0], sola=[-x,0,0], yukarı/havaya=[0,+y,0], aşağı/dibe=[0,-y,0], ileri=[0,0,+z], geri=[0,0,-z].
- Miktar belirtilmemişse varsayılan adımı 2 birim al.
- Olumsuz komut verilen nesnelere ("sakın oynatma", "elleme") ASLA action üretme.
- Yanıtında JSON dışında HİÇBİR açıklama yazma.
"""

def compare_actions(expected_list: list, actual_list: list) -> tuple[bool, bool]:
    """(target_match, vector_match) döndürür."""
    if not actual_list or len(expected_list) != len(actual_list):
        return False, False

    exp_sorted = sorted(expected_list, key=lambda x: x["target"])
    act_sorted = sorted(actual_list, key=lambda x: str(x.get("target", "")))

    target_ok = True
    vector_ok = True

    for exp, act in zip(exp_sorted, act_sorted):
        if exp["target"].lower() != str(act.get("target", "")).lower():
            target_ok = False
        act_pos = act.get("pos", [])
        if len(act_pos) != 3 or any(abs(float(e) - float(a)) > 0.01 for e, a in zip(exp["pos"], act_pos)):
            vector_ok = False

    return target_ok, vector_ok

def check_semantic_keywords(text: str, keywords: list) -> tuple[bool, list]:
    """Üretilen Türkçe soru/gerekçe metninde beklenen anlamsal anahtar kelimeleri denetler."""
    if not text or not keywords:
        return False, []
    text_lower = text.lower()
    matched = [kw for kw in keywords if kw.lower() in text_lower]
    return len(matched) > 0, matched

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "eval_dataset.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scene_state = data["scene_state"]
    test_cases = data["test_cases"]

    total_decision_ok = 0
    action_total = 0
    target_ok_count = 0
    vector_ok_count = 0
    full_action_ok_count = 0
    ask_user_ok_count = 0
    reject_ok_count = 0
    audit_logs = []

    print("=" * 85)
    print("🔍 KATMAN 2: UÇTAN UCA PARAMETRE (SLOT-FILLING) & SEMANTİK YANIT DENETİMİ")
    print("=" * 85)

    for item in test_cases:
        cmd_id = item["id"]
        cmd = item["command"]
        exp_type = item["expected_type"]

        try:
            res = client.models.generate_content(
                model=MODEL_NAME,
                contents=cmd,
                config=types.GenerateContentConfig(
                    system_instruction=build_production_system_instruction(scene_state),
                    response_mime_type="application/json",
                    temperature=0.0
                ),
            )
            parsed = json.loads(res.text)
        except Exception as e:
            parsed = {"type": "error", "error": str(e)}

        act_type = parsed.get("type", "unknown")
        decision_ok = (act_type == exp_type)
        if decision_ok:
            total_decision_ok += 1

        detail_note = ""
        slot_status = "N/A"
        matched_kws = []

        if exp_type == "action":
            action_total += 1
            exp_acts = item.get("expected_actions", [])
            act_acts = parsed.get("actions", []) if act_type == "action" else []
            t_ok, v_ok = compare_actions(exp_acts, act_acts)
            if t_ok:
                target_ok_count += 1
            if v_ok:
                vector_ok_count += 1
            if decision_ok and t_ok and v_ok:
                full_action_ok_count += 1
                slot_status = "✅ TAM VEKTÖR & HEDEF EŞLEŞMESİ"
            else:
                slot_status = f"❌ HATA (Beklenen: {exp_acts} | Gelen: {act_acts})"
            detail_note = f"Actions: {act_acts}"

        elif exp_type == "ask_user":
            q_text = parsed.get("question", "")
            kw_ok, matched_kws = check_semantic_keywords(q_text, item.get("expected_keywords", []))
            if decision_ok and kw_ok:
                ask_user_ok_count += 1
                slot_status = f"✅ GEÇERLİ SORU (Anahtar: {matched_kws})"
            else:
                slot_status = f"❌ EKSİK İÇERİK (Beklenen: {item.get('expected_keywords')})"
            detail_note = f"Soru: '{q_text}'"

        elif exp_type == "reject_command":
            r_text = parsed.get("reason", "")
            kw_ok, matched_kws = check_semantic_keywords(r_text, item.get("expected_keywords", []))
            if decision_ok and kw_ok:
                reject_ok_count += 1
                slot_status = f"✅ GEÇERLİ GEREKÇE (Anahtar: {matched_kws})"
            else:
                slot_status = f"❌ EKSİK İÇERİK (Beklenen: {item.get('expected_keywords')})"
            detail_note = f"Gerekçe: '{r_text}'"

        print(f"[{cmd_id:02d}/75] '{cmd}'")
        print(f"        Sınıf: {act_type:<14} (Beklenen: {exp_type:<14}) | {slot_status}")
        print(f"        Detay: {detail_note}")

        audit_logs.append({
            "id": cmd_id,
            "command": cmd,
            "expected_type": exp_type,
            "actual_type": act_type,
            "decision_correct": decision_ok,
            "slot_status": slot_status,
            "matched_keywords": matched_kws,
            "raw_output": parsed
        })

        time.sleep(1.2)

    print("\n" + "=" * 85)
    print("📊 UÇTAN UCA PARAMETRE (SLOT-FILLING) & SEMANTİK YANIT DENETİM ÖZETİ")
    print("=" * 85)
    print(f"1. Üst Karar Sınıfı Doğruluğu (Decision Accuracy)         : %{total_decision_ok/len(test_cases)*100:.2f} ({total_decision_ok}/{len(test_cases)})")
    print(f"2. Hedef Nesne Doğruluğu (Target Entity Accuracy)         : %{target_ok_count/action_total*100:.2f} ({target_ok_count}/{action_total})")
    print(f"3. 3B Hareket Vektörü Doğruluğu (3D Offset Accuracy)      : %{vector_ok_count/action_total*100:.2f} ({vector_ok_count}/{action_total})")
    print(f"4. Tam Eylem Eşleşmesi (Exact Action Execution Match)     : %{full_action_ok_count/action_total*100:.2f} ({full_action_ok_count}/{action_total})")
    print(f"5. Semantik Soru Kalitesi (Clarification Keyword Match)   : %{ask_user_ok_count/25*100:.2f} ({ask_user_ok_count}/25)")
    print(f"6. Semantik Reddetme Kalitesi (Refusal Keyword Match)     : %{reject_ok_count/25*100:.2f} ({reject_ok_count}/25)")
    print("=" * 85)

    out_path = os.path.join(base_dir, "execution_audit_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME,
            "summary": {
                "decision_accuracy": round(total_decision_ok / len(test_cases), 4),
                "target_entity_accuracy": round(target_ok_count / action_total, 4),
                "vector_offset_accuracy": round(vector_ok_count / action_total, 4),
                "exact_action_match": round(full_action_ok_count / action_total, 4),
                "clarification_semantic_quality": round(ask_user_ok_count / 25, 4),
                "refusal_semantic_quality": round(reject_ok_count / 25, 4)
            },
            "logs": audit_logs
        }, f, ensure_ascii=False, indent=2)
    print(f"📄 Tam denetim raporu kaydedildi: {out_path}")

if __name__ == "__main__":
    main()