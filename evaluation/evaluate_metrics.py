import json
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. KONFİGÜRASYON
# ---------------------------------------------------------
GEMINI_API_KEY = "BURAYA_API_KEY_YAZIN"  # <-- Kendi API anahtarını yapıştır
MODEL_NAME = "gemini-3.8-flash"  # Kullandığın model adını buraya yaz (örn: gemini-2.5-flash veya gemini-2.0-flash)

client = genai.Client(api_key=GEMINI_API_KEY)
CLASSES = ["action", "ask_user", "reject_command"]

# ---------------------------------------------------------
# 2. DEĞERLENDİRME SYSTEM PROMPT (GÜVEN SKORU DESTEKLİ)
# ---------------------------------------------------------
def build_eval_system_instruction(scene_state: dict) -> str:
    objects_summary = json.dumps(scene_state.get("objects", []), ensure_ascii=False)
    
    return f"""
Sen 3B bir sanal ortamda nesneleri yöneten akıllı bir yapay zeka ajanısın.
Şu anki sahnede bulunan nesneler, renkleri ve mevcut konumları şunlardır:
{objects_summary}

Kullanıcının Türkçe doğal dil komutlarını analiz et ve SADECE aşağıdaki JSON formatını üret:

{{
  "type": "action" | "ask_user" | "reject_command",
  "confidence_scores": {{
    "action": 0.0 ile 1.0 arası olasılık değeri,
    "ask_user": 0.0 ile 1.0 arası olasılık değeri,
    "reject_command": 0.0 ile 1.0 arası olasılık değeri
  }},
  "payload": "Eylem özeti, netleştirme sorusu veya reddetme gerekçesi"
}}

SINIFLANDIRMA KURALLARI:
1. "action": Komut sahnedeki mevcut bir nesneyi hedef alıyorsa, hangi nesne olduğu tekil olarak ayırt edilebiliyorsa (örn. renk veya benzersiz tür ile) ve hareket yönü belliyse seçilir. (Miktar belirtilmemişse varsayılan 2 birim alınır, bu bir belirsizlik değildir).
2. "ask_user": Sahnede aynı türden birden fazla nesne varken hangisi olduğu belirtilmemişse (örn. 2 küp varken sadece "küpü kaydır" denmişse), işaret zamiri ("onu", "şunu") kullanılmışsa veya hareket yönü hiç belirtilmemişse ("küreyi hareket ettir") seçilir.
3. "reject_command": Sahnede olmayan bir nesne veya yanlış renk-nesne eşleşmesi ("yeşil küp", "masa") istendiyse, fizik dışı/desteklenmeyen bir eylem ("patlat", "ikiye böl", "rengini değiştir", "duvarın içine sok") veya mantıksal çelişki ("kıpırdatmadan ileri götür") varsa seçilir.
4. "confidence_scores" içindeki üç değerin toplamı 1.0 olmalıdır. Seçtiğin "type" en yüksek skora sahip olmalıdır.
5. Yanıtında JSON dışında HİÇBİR açıklama yazma.
"""

def evaluate_single_command(user_prompt: str, scene_state: dict) -> dict:
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=build_eval_system_instruction(scene_state),
                response_mime_type="application/json",
                temperature=0.0  # Deterministik sonuçlar için 0.0
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"   [API Hatası]: {e}")
        return None

# ---------------------------------------------------------
# 3. ANA TEST VE METRİK HESAPLAMA AKIŞI
# ---------------------------------------------------------
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "eval_dataset.json")

    if not os.path.exists(dataset_path):
        print(f"HATA: {dataset_path} bulunamadı!")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scene_state = data["scene_state"]
    test_cases = data["test_cases"]

    # Bilimsel tarafsızlık için sabit seed ile karıştırma (Shuffle)
    random.seed(42)
    random.shuffle(test_cases)

    y_true = []
    y_pred = []
    y_scores = []
    detailed_logs = []

    print("=" * 75)
    print(f"🚀 ARA RAPOR DEĞERLENDİRME TESTİ BAŞLATILIYOR ({len(test_cases)} Senaryo)")
    print(f"🤖 Model: {MODEL_NAME} | Temperature: 0.0")
    print("=" * 75)

    for idx, item in enumerate(test_cases, start=1):
        cmd_id = item["id"]
        cmd_text = item["command"]
        expected = item["expected_type"]
        category = item.get("category", "")

        result = evaluate_single_command(cmd_text, scene_state)

        if result and result.get("type") in CLASSES:
            pred_type = result["type"]
            scores_dict = result.get("confidence_scores", {})
            # Sınıf sırasına göre olasılık vektörü oluştur ve normalize et
            raw_probs = [float(scores_dict.get(c, 0.0)) for c in CLASSES]
            total_p = sum(raw_probs)
            if total_p > 0:
                probs = [p / total_p for p in raw_probs]
            else:
                probs = [1.0 if c == pred_type else 0.0 for c in CLASSES]
            payload = result.get("payload", "")
        else:
            pred_type = "reject_command"  # Fallback
            probs = [0.0, 0.0, 1.0]
            payload = "Parse/API Error"

        y_true.append(expected)
        y_pred.append(pred_type)
        y_scores.append(probs)

        is_correct = (expected == pred_type)
        status_icon = "✅" if is_correct else "❌"
        print(f"[{idx:02d}/{len(test_cases)}] (ID:{cmd_id:02d}) '{cmd_text}'")
        print(f"        Beklenen: {expected:<14} | Tahmin: {pred_type:<14} {status_icon} | Skorlar: {[round(p, 2) for p in probs]}")

        detailed_logs.append({
            "id": cmd_id,
            "command": cmd_text,
            "category": category,
            "expected": expected,
            "predicted": pred_type,
            "correct": is_correct,
            "confidence_scores": dict(zip(CLASSES, [round(p, 4) for p in probs])),
            "payload": payload
        })

        # Ücretsiz API Rate Limit koruması
        time.sleep(1.2)

    # ---------------------------------------------------------
    # 4. METRİK HESAPLAMALARI
    # ---------------------------------------------------------
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_scores_arr = np.array(y_scores)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="weighted", zero_division=0
    )

    print("\n" + "=" * 75)
    print("📊 SINIFLANDIRMA PERFORMANS METRİKLERİ")
    print("=" * 75)
    print(f"Genel Doğruluk (Accuracy)       : %{acc * 100:.2f} ({sum(y_true_arr == y_pred_arr)}/{len(test_cases)})")
    print(f"Macro Precision (Kesinlik)      : %{prec_macro * 100:.2f}")
    print(f"Macro Recall (Duyarlılık)       : %{rec_macro * 100:.2f}")
    print(f"Macro F1-Score                  : %{f1_macro * 100:.2f}")
    print("-" * 75)
    print("\nSınıf Bazlı Detaylı Rapor:\n")
    print(classification_report(y_true_arr, y_pred_arr, target_names=CLASSES, digits=4, zero_division=0))

    # ---------------------------------------------------------
    # 5. KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX) GRAFİĞİ
    # ---------------------------------------------------------
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=CLASSES)
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("3B LLM Ajanı - Karar Mekanizması Karmaşıklık Matrisi\n(Confusion Matrix - 60 Senaryo)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Tahmin Edilen Sınıf (Predicted Label)", fontsize=11, fontweight="bold")
    plt.ylabel("Gerçek Sınıf (True Label)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(base_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"🖼️ Karmaşıklık Matrisi kaydedildi: {cm_path}")

    # ---------------------------------------------------------
    # 6. ÇOKLU SINIF ROC EĞRİSİ (ONE-VS-REST ROC-AUC)
    # ---------------------------------------------------------
    y_true_bin = label_binarize(y_true_arr, classes=CLASSES)
    n_classes = len(CLASSES)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores_arr[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8, 6), dpi=300)
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2.2,
            label=f"Sınıf: {CLASSES[i]} (AUC = {roc_auc[i]:.4f})"
        )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        color="#2980b9",
        linestyle="--",
        lw=2.5,
        label=f"Macro-Average ROC (AUC = {roc_auc['macro']:.4f})"
    )
    plt.plot([0, 1], [0, 1], "k:", lw=1.5, label="Rastgele Sınıflandırıcı (AUC = 0.5000)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Yanlış Pozitif Oranı (False Positive Rate - FPR)", fontsize=11, fontweight="bold")
    plt.ylabel("Doğru Pozitif Oranı (True Positive Rate - TPR)", fontsize=11, fontweight="bold")
    plt.title("Çoklu Sınıf ROC Eğrisi (One-vs-Rest ROC-AUC)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(base_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"📈 ROC Eğrisi kaydedildi: {roc_path}")

    # ---------------------------------------------------------
    # 7. SONUÇLARI JSON OLARAK KAYDET
    # ---------------------------------------------------------
    report_data = {
        "model": MODEL_NAME,
        "total_samples": len(test_cases),
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_precision": round(prec_macro, 4),
            "macro_recall": round(rec_macro, 4),
            "macro_f1": round(f1_macro, 4),
            "weighted_f1": round(f1_weighted, 4),
            "roc_auc_per_class": {CLASSES[i]: round(roc_auc[i], 4) for i in range(n_classes)},
            "macro_roc_auc": round(roc_auc["macro"], 4)
        },
        "confusion_matrix": {
            "labels": CLASSES,
            "matrix": cm.tolist()
        },
        "detailed_results": detailed_logs
    }

    report_path = os.path.join(base_dir, "results_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"📄 Detaylı rapor JSON kaydedildi: {report_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()