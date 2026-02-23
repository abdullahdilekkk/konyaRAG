import os
import requests
from config.settings import LLM_MODEL_NAME

# DİKKAT ÖĞRETİCİ NOT:
# RAG sistemlerinde "Generation (Üretme)" adımı, arabulucu (sekreter) gibidir.
# Önceki dosyada bulduğu metinleri ve kullanıcının sorusunu birleştirerek
# "Lütfen bu metinlere bakarak bu soruya cevap ver" diyen büyük bir metin (Prompt) hazırlar.
# Ardından bunu ChatGPT, LLaMA veya Claude gibi bir modele yollar.

def secilen_metinler_ile_cevap_uret(kullanici_sorusu: str, bulunan_metinler: list):
    """
    Bu fonksiyon:
    1. Kullanıcının sorusunu alır.
    2. Milvus'tan (retrieval.py) dönen alakalı metinleri alır.
    3. Hepsini birleştirip "Prompt" adı verilen bir emir kâğıdına dönüştürür.
    4. Bu emri Büyük Dil Modeline (LLM - yapay zekaya) yollar.
    5. Yapay zekanın "insan gibi" verdiği Türkçe cevabı ekrana döndürür.
    """
    
    # Eğer Milvus hiçbir metin bulamadıysa boşuna yapay zekayı yormayalım.
    if not bulunan_metinler:
        return "Üzgünüm, soruna dair veritabanımda hiçbir bilgi bulamadım."
        
    # =========================================================
    # Milvus'tan dönen liste (bulunan_metinler) içindeki tüm metinleri aralarına boşluk koyarak
    # tek bir devasa "bilgi metni" haline getiriyoruz.
    bilgi_yakiti = "\n".join(bulunan_metinler)
    
    # SENİOR DEBUG (HATA AYIKLAMA) NOTU:
    # Yapay zekaya gitmeden önce, veritabanından cımbızladığımız o 3 metin parçası acaba neymiş?
    # Gerçekten içinde etliekmek geçiyor muymuş? Kendimiz okuyalım diye ekrana basıyoruz:
    print("\n--- 🔍 MİLVUS'TAN GELEN HAM BİLGİ METNİ (LLM'in Okuduğu) ---")
    print(bilgi_yakiti)
    print("----------------------------------------------------------\n")
    
    # Modelin görevi yanlış anlamaması için ona sert bir kural (Prompt) yazıyoruz:
    emir_kagidi = f"""
    Sen, sadece sana verilen 'BİLGİ' metnine dayanarak soruları cevaplayan akıllı bir asistansın.
    Asla BİLGİ kısmında geçmeyen bir yeteneği veya yorumu uydurma. Bilgi yoksa "Bilmiyorum" de.
    
    --- BİLGİ BAŞLANGICI ---
    {bilgi_yakiti}
    --- BİLGİ BİTİŞİ ---
    
    Soru: {kullanici_sorusu}
    
    Cevap:
    """
    
    # 1. KAPI (Endpoint): İstek atacağımız URL adresi.
    OLLAMA_URL = "http://localhost:11434/api/generate"
    
    payload = {
        "model": LLM_MODEL_NAME, # Hangi modeli kullanacağız? (Örn: "llama3", "mistral" veya settings'teki model)
        "prompt": emir_kagidi,
        "stream": False # Cevabı kelime kelime yavaşça değil, tamamen bitince tek seferde istiyoruz (False)
    }
    
    # Ollama'ya kendi bilgisayarımızdaki (localhost) sistemden JSON formatında istek atıyoruz:
    gelen_cevap = requests.post(OLLAMA_URL, json=payload)
    
    # OLLAMA BİR HATA DÖNDÜRDÜYSE (Model yok, silinmiş veya port yanlışsa)
    if gelen_cevap.status_code != 200:
        return f"OLLAMA HATASI! (Kod: {gelen_cevap.status_code}) -> {gelen_cevap.text}"
    
    # 4. CEVABI AÇ (Response Parsing):
    # Kapıdan dönen kargoyu önce bilgisayarın anladığı dilden JSON sözlüğüne çeviriyoruz
    sonuc_json = gelen_cevap.json()
    
    # O sözlüğün içinden asıl Türkçe cümlenin yazdığı "response" etiketini cımbızla çekiyoruz
    olusan_cevap = sonuc_json.get("response", "HATA: Ollama'dan başarılı döndü ama içi boş.")
    
    return olusan_cevap.strip()


if __name__ == "__main__":
    # Üst klasördeki 'retrieval' dosyasından arama fonksiyonumuzu içeriye dahil ediyoruz (İthal ediyoruz)
    from retrieval import soruyu_milvusta_ara 
    
    print("=====================================================")
    print(" KONYA RAG SİSTEMİNE HOŞGELDİNİZ (Test Modu)")
    print(" Çıkmak için 'q' tuşuna basıp Enter'a basabilirsiniz.")
    print("=====================================================\n")
    
    while True:
        # 1. Sisteme sormak istediğimiz soruyu artık kodun içine yazmıyoruz,
        # Klavyeden (Terminalden) dinamik olarak o an ne sormak istiyorsak onu alıyoruz:
        kralin_sorusu = input("Lütfen sorunuzu girin: ")
        
        # Eğer çıkmak istersek q yazıp çıkarız
        if kralin_sorusu.lower() == 'q':
            print("Sistemden çıkılıyor. Görüşmek üzere!")
            break
            
        print("\n⏳ Milvus Veritabanında (Hafızada) eşleşen parçalar aranıyor...")
        
        # 2. Hafızadaki (Milvus'taki) ilgili PDF parçacıklarını arayıp buluyoruz
        bulunan_parcalar = soruyu_milvusta_ara(kralin_sorusu, kac_cevap_getirsin=3)
        
        # 3. Bulunan bu parçaları ve kullanıcının girdiği soruyu Yapay Zekaya gönderiyoruz
        if bulunan_parcalar:
             print(f"Milvus'tan {len(bulunan_parcalar)} adet metin parçası bulundu.")
             print("Şimdi Yapay Zeka (Ollama) cümleyi toparlıyor...\n")
             
             nihai_cevap = secilen_metinler_ile_cevap_uret(kullanici_sorusu=kralin_sorusu, bulunan_metinler=bulunan_parcalar)
             
             print("--- OLLAMA'NIN CEVABI ---")
             print(nihai_cevap)
             print("------------------------------------------------\n")
        else:
             print("Milvus'ta hiçbir parça bulunamadı! Lütfen önce veritabanının dolu olduğundan emin ol.\n")