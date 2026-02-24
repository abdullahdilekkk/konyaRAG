import os
import re #regex kütüphanesi
import fitz  # PyMuPDF (PDF Okuyucu)
from sentence_transformers import SentenceTransformer
from core.database import connectClient, initClient
from config.settings import COLLECTION_NAME, EMBEDDING_MODEL_NAME

import pandas as pd # Excel Okuyucu

def pdf_verilerini_veritabanina_yukle(pdf_klasoru):
    """
    Bu fonksiyon:
    1. İlgili dosyaları harf harf okur (Extract).
    2. Okunan devasa metni anlamı bozulmayacak şekilde küçük parçalara (Chunk) ayırır.
    3. Bu parçaları yapay zekanın anlayacağı sayılara (Vektörlere / Embeddings) dönüştürür.
    4. Tüm bu sayıları ve metinleri Milvus veritabanımıza kaydeder.
    """
    
    # HAZIRLIK: Veritabanına bağlan ve AI Modelimizi yükle
    initClient()
    
    client = connectClient()
    
    ai_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Klasördeki tüm dosyaları tek tek gez:
    for dosya_adi in os.listdir(pdf_klasoru):
        if not (dosya_adi.endswith(".pdf") or dosya_adi.endswith(".txt") or dosya_adi.endswith(".xlsx") or dosya_adi.endswith(".xls")):
            continue # İzin verilen formatlardan değilse diğer dosyaya geç atla.
            
        dosya_yolu = os.path.join(pdf_klasoru, dosya_adi)
        print(f"---> İŞLENİYOR: {dosya_adi}")
        
        # ADIM 1: DOSYAYI OKUMA (Extract)
        tum_metin = ""
        
        if dosya_adi.endswith(".pdf"):
            pdf_belgesi = fitz.open(dosya_yolu)
            for sayfa in pdf_belgesi:
                tum_metin += sayfa.get_text() + "\n"
        elif dosya_adi.endswith(".txt"):
            with open(dosya_yolu, 'r', encoding='utf-8') as f:
                tum_metin = f.read()
        elif dosya_adi.endswith(".xlsx") or dosya_adi.endswith(".xls"):
            # Excel dosyalarını 'pandas' tablosuna (DataFrame) alıyoruz.
            df = pd.read_excel(dosya_yolu)
            df = df.fillna("") # Tablodaki boş hücreleri 'NaN' yerine temiz boşluk yap
            # Tabloyu baştan aşağı metne (String) dönüştür
            tum_metin = df.to_string(index=False)
            
        # --- AĞIR TEMİZLİK (ADVANCED DATA CLEANING) BÖLÜMÜ ---
        # 1. Her türlü URL, link ve '.com', '.tr', 'php?' gibi web artıklarını sil
        tum_metin = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', tum_metin)
        tum_metin = re.sub(r'www\.\S+', ' ', tum_metin)
        tum_metin = re.sub(r'[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/\S*)?', ' ', tum_metin) # Kalan yarım linkleri de sil ('konya.bel.tr' gibi)
        tum_metin = re.sub(r'\b\w+\.php\?\S+', ' ', tum_metin) # 'detail.php?id=' tarzı pislikleri sil
        
        # 2. Wikipedia'nın "51. ^ ", "12. ^" tarzındaki kaynakça ve dipnot çer çöpünü uçur
        tum_metin = re.sub(r'\d+\.\s+\^\s+', ' ', tum_metin)
        tum_metin = re.sub(r'\[\d+\]', ' ', tum_metin)
        
        # 3. Kalan tuhaf parantezli kopuk heceleri ( p?haberID=... ) sil
        tum_metin = re.sub(r'\(\s*[a-zA-Z0-9_\-\.\?\=\/]*\s*\)', ' ', tum_metin)
        
        # 4. "Arşivlenmiş kopya" veya "Erişim tarihi" gereksiz kelimeleri sil
        tum_metin = tum_metin.replace("Arşivlenmiş kopya", " ")
        tum_metin = re.sub(r'Erişim tarihi:\s*\d{1,2}\s+[A-Za-zÇŞĞÜÖİçşğüöı]+\s+\d{4}', ' ', tum_metin)
        
        # 5. Son ütü: Çift/Üçlü boşlukları teke indir ve satır başlıklarını temizle
        tum_metin = re.sub(r'\s+', ' ', tum_metin).strip()
        # ----------------------------------------------
        
        # ADIM 2: METNİ MANTIKLI PARÇALARA BÖLME (Chunking) 
        cumleler = re.split(r'(?<=[.!?])\s+|\n+', tum_metin)
        
        parcalar = []       # En son elimizde kalacak temiz parçalar
        gecici_sepet = ""   # Cümleleri biriktirdiğimiz geçici kutu
        LIMIT = 800         # Bir parça en fazla 800 karakter olabilir
        
        for cumle in cumleler:
            cumle = cumle.strip()
            if not cumle: continue
            
            # Eğer sepetteki mecut yazıya bu yeni cümleyi de eklersek 800'ü taşıyor mu?
            if len(gecici_sepet) + len(cumle) > LIMIT and len(gecici_sepet) > 0:
                # Taşıyorsa sepet dolmuş demektir, ana listeye at.
                parcalar.append(gecici_sepet.strip())
                
                # Yeni sepete başlarken önceki cümlenin son 150 harfini de alıyoruz ki (Overlap) bağlam kopmasın.
                kalan_baglam = gecici_sepet[-150:] if len(gecici_sepet) > 150 else gecici_sepet
                gecici_sepet = kalan_baglam + " " + cumle + " "
            else:
                # Daha limite varmadıysak cümleyi sepete atmaya devam et.
                gecici_sepet += cumle + " "
                
        # İçeride kalan son yarım sepeti de unutma!
        if gecici_sepet:
            parcalar.append(gecici_sepet.strip())
            
        print(f"PDF toplam {len(parcalar)} adet metin parçasına bölündü.")
        
        # =========================================================
        # ADIM 3: PARÇALARI VEKTÖRE (SAYILARA) ÇEVİRME (Embedding)
        # =========================================================
        # DİKKAT ÖĞRETİCİ NOT:
        # ai_model.encode() fonksiyonu içeriye bir liste (parcalar) aldığında, 
        # sana zaten o kadar elemanlı "listelerin listesini" (vektörleri) geri verir.
        # O YÜZDEN ASLA 'parcalar[0]' veya 'vektorler[0]' YAPMIYORUZ! 
        # Çünkü PDF tek bir cümle değil, yüzlerce parçadır. Bize hepsi lazım.
        
        dizi_vektorler = ai_model.encode(parcalar)
        saf_vektor_listesi = dizi_vektorler.tolist() # Numpy dizisini saf Python listesine çevirdik.

        
        # =========================================================
        # ADIM 4: HER ŞEYİ BİRLEŞTİRİP VERİTABANINA YAZMA 
        # =========================================================
        kaydedilecek_veriler = []
        # Hem metnin kendisini ("Konya çok büyüktür") hem de yapay zekanın sayısal vektörünü [0.45, -0.12] paket yapacağız.
        for i in range(len(parcalar)):
            kaydedilecek_veriler.append({
                "text": parcalar[i],
                "vector": saf_vektor_listesi[i]
            })
            
        client.insert(collection_name=COLLECTION_NAME, data=kaydedilecek_veriler)
        print(f"BAŞARILI: {dosya_adi} tamamen Milvus'a kaydedildi!\n")


if __name__ == "__main__":
    import pathlib
    # Önceki kodda hata veren dosya yolunu (os.path...) Python'un modern Pathlib kütüphanesiyle çözdük.
    # Bu satır, bu python dosyasının olduğu yerden (services klasörü) 2 tık geriye çıkıp "data" klasörünü bulur.
    data_klasoru = pathlib.Path(__file__).parent.parent / "data"
    
    if data_klasoru.exists():
        pdf_verilerini_veritabanina_yukle(str(data_klasoru))
    else:
        print(f"HATA: Veri klasörü bulunamadı -> {data_klasoru}")
