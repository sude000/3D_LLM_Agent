import os
import re
import shutil
import numpy as np

# imageio-ffmpeg ikilisini .venv/Scripts/ffmpeg.exe olarak kopyala ve PATH'e ekle
try:
    import imageio_ffmpeg
    src_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(src_ffmpeg)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    venv_scripts = os.path.join(os.getcwd(), ".venv", "Scripts")
    target_ffmpeg = os.path.join(venv_scripts, "ffmpeg.exe")
    if os.path.isdir(venv_scripts) and not os.path.exists(target_ffmpeg):
        shutil.copy2(src_ffmpeg, target_ffmpeg)
    if os.path.isdir(venv_scripts):
        os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")
except Exception as e:
    print(f"[STT Adaptörü Uyarı] ffmpeg ayarlanırken uyarı: {e}")

_whisper_model = None
WHISPER_AVAILABLE = True

# Whisper'a sahne kelimelerini ve "birim" kalıbını öğreten bağlam ipucu
DOMAIN_INITIAL_PROMPT = (
    "3B sanal ortam nesne kontrol komutları: kırmızı küp, mavi küp, yeşil küre, "
    "sarı silindir, silindiri, küpü, küreyi, 1 birim, 2 birim, 3 birim, 4 birim, 5 birim, "
    "sağa, sola, yukarı, aşağı, ileri, geri kaydır, kaldır, indir, götür, it, çek."
)


def _load_audio_without_external_ffmpeg(file: str, sr: int = 16000) -> np.ndarray:
    """Ses dosyasını doğrudan 16 kHz mono float32 numpy dizisine çevirir."""
    import subprocess
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def get_whisper_model():
    """Türkçe başarımı yüksek olan OpenAI Whisper ('small') modelini yükler."""
    global _whisper_model, WHISPER_AVAILABLE
    if _whisper_model is not None:
        return _whisper_model

    try:
        import whisper
        import whisper.audio

        whisper.audio.load_audio = _load_audio_without_external_ffmpeg

        print("[STT Adaptörü] OpenAI Whisper ('small') modeli yükleniyor...")
        _whisper_model = whisper.load_model("small", device="cpu")
        WHISPER_AVAILABLE = True
        print("[STT Adaptörü] OpenAI Whisper ('small') modeli başarıyla hazırlandı.")
        return _whisper_model
    except Exception as e:
        WHISPER_AVAILABLE = False
        print(f"[STT Adaptörü Uyarı] Whisper başlatılamadı (Yazılı mod aktif): {e}")
        return None


def turkish_lower(text: str) -> str:
    """Türkçe karakter duyarlı küçük harf dönüşümü (İ -> i, I -> ı)."""
    text = text.replace("İ", "i").replace("I", "ı")
    return text.lower()


def normalize_turkish_text(raw_text: str) -> str:
    """
    Türkçe metin normalizasyonu:
    - Türkçe karakter uyumlu küçük harf dönüşümü
    - Dolgu kelimeleri temizleme
    - Yazıyla yazılan sayıları rakama çevirme
    - "birim" -> "1" karışıklığını ve nesne/fiil fonetik kaymalarını düzeltme
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    cleaned = turkish_lower(raw_text.strip())
    cleaned = re.sub(r"[^\w\s.,-]", " ", cleaned)

    # 1. Konuşma dolgu kelimelerini temizle
    filler_words = [
        r"\bı+\b", r"\be+\b", r"\ba+\b",
        r"\bşey\b", r"\byani\b", r"\bişte\b",
        r"\bhani\b", r"\bfalan\b", r"\bfilan\b",
        r"\bmesela\b", r"\blütfen\b"
    ]
    for pattern in filler_words:
        cleaned = re.sub(pattern, " ", cleaned)

    # 2. Yazıyla algılanan sayıları rakama dönüştür
    number_map = {
        r"\bbir\b": "1",
        r"\biki\b": "2",
        r"\büç\b": "3",
        r"\bdört\b": "4",
        r"\bbeş\b": "5",
        r"\baltı\b": "6",
        r"\byedi\b": "7",
        r"\bsekiz\b": "8",
        r"\bdokuz\b": "9",
        r"\bon\b": "10",
    }
    for word_pattern, digit in number_map.items():
        cleaned = re.sub(word_pattern, digit, cleaned)

    # 3. "birim" kelimesinin "1" veya birleşik yazılma hatalarını düzelt
    cleaned = re.sub(r"\bkibirim\b", "2 birim", cleaned)
    cleaned = re.sub(r"\b1im\b", "birim", cleaned)
    cleaned = re.sub(r"\b1\s*im\b", "birim", cleaned)
    cleaned = re.sub(r"\b(\d+)\s+1\b", r"\1 birim", cleaned)
    cleaned = re.sub(r"\b(\d+)\s+1\s+(sağa|sola|yukarı|aşağı|ileri|geri)\b", r"\1 birim \2", cleaned)

    # 4. Sık görülen Türkçe STT nesne/fiil fonetik hatalarını düzelt
    phonetic_fixes = {
        r"\bselindere\b": "silindiri",
        r"\bselindir\b": "silindir",
        r"\bsilindere\b": "silindiri",
        r"\bsilindir\s+in\b": "silindiri",
        r"\bsilindir\s+i\b": "silindiri",
        r"\bkupü\b": "küpü",
        r"\bkupu\b": "küpü",
        r"\bkup\b": "küp",
        r"\bküp\s+ü\b": "küpü",
        r"\bkureyi\b": "küreyi",
        r"\bkure\b": "küre",
        r"\bküre\s+yi\b": "küreyi",
        r"\bkalır\b": "kaldır",
    }
    for wrong_pattern, correct_word in phonetic_fixes.items():
        cleaned = re.sub(wrong_pattern, correct_word, cleaned)

    return re.sub(r"\s+", " ", cleaned).strip(" .,")


def transcribe_audio(audio_filepath: str) -> tuple[bool, str]:
    """
    Resmi OpenAI Whisper ('small' + domain prompt) ile sesi Türkçe metne çevirir.
    """
    if not audio_filepath or not os.path.exists(audio_filepath):
        return False, "Ses kaydı bulunamadı. Kaydı durdurup 1 saniye bekledikten sonra gönderin."

    model = get_whisper_model()
    if model is None:
        return False, "Ses modülü şu an devre dışı (Bağlantı korundu). Yazılı olarak devam edebilirsiniz."

    try:
        audio_array = _load_audio_without_external_ffmpeg(audio_filepath)
        result = model.transcribe(
            audio_array,
            language="tr",
            initial_prompt=DOMAIN_INITIAL_PROMPT,
            temperature=0.0,
            fp16=False
        )
        raw_transcript = result.get("text", "").strip()
        print(f"[STT Ham Çıktı] '{raw_transcript}'")

        if not raw_transcript:
            return False, "Seste anlaşılır bir konuşma tespit edilemedi."

        normalized = normalize_turkish_text(raw_transcript)
        print(f"[STT Normalize] '{normalized}'")

        if not normalized:
            return False, "Sadece dolgu kelimeler algılandı, geçerli bir komut bulunamadı."

        return True, normalized

    except Exception as e:
        print(f"[STT Adaptörü Hata] {e}")
        return False, "Ses işlenirken geçici bir hata oluştu (Bağlantı korundu). Yazılı olarak devam edebilirsiniz."