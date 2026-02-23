import os
import re
import fitz  # PyMuPDF (PDF Okuyucu)
from sentence_transformers import SentenceTransformer
from core.database import connectClient
from config.settings import COLLECTION_NAME, EMBEDDING_MODEL_NAME

def pdf_verilerini_veritabanina_yukle(pdf_klasoru):
    """
    Bu fonksiyon:
    1. PDF'leri açar ve harf harf okur (Extract).
    2. Okunan devasa metni anlamı bozulmayacak şekilde küçük parçalara (Chunk) ayırır.
    3. Bu parçaları yapay zekanın anlayacağı sayılara (Vektörlere / Embeddings) dönüştürür.
    4. Tüm bu sayıları ve metinleri Milvus veritabanımıza kaydeder.
    """
    
    # ---------------------------------------------------------
    # HAZIRLIK: Veritabanına bağlan ve AI Modelimizi yükle
    # ---------------------------------------------------------
    client = connectClient()
    
    ai_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Klasördeki tüm dosyaları tek tek gez:
    for dosya_adi in os.listdir(pdf_klasoru):
        if not dosya_adi.endswith(".pdf"):
            continue # PDF değilse diğer dosyaya geç atla.
            
        dosya_yolu = os.path.join(pdf_klasoru, dosya_adi)
        print(f"---> İŞLENİYOR: {dosya_adi}")
        
        # =========================================================
        # ADIM 1: PDF DOSYASINI OKUMA (Extract)
        # =========================================================
        pdf_belgesi = fitz.open(dosya_yolu)
        tum_metin = ""
        for sayfa in pdf_belgesi:
            tum_metin += sayfa.get_text() + "\n"
        
        # =========================================================
        # ADIM 2: METNİ MANTIKLI PARÇALARA BÖLME (Chunking) 
        # Neden bölüyoruz? Çünkü AI Modeli tek seferde koskoca PDF'i anlayamaz. 
        # Cümleleri tam ortadan kesmemek için Nokta(.) falan görünce böleceğiz.
        # =========================================================
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
