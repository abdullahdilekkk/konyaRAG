from sentence_transformers import SentenceTransformer
from core.database import connectClient
from config.settings import COLLECTION_NAME, EMBEDDING_MODEL_NAME

# Modeli her soruda baştan yüklememek için (HIZ İÇİN) Global hafızaya alıyoruz.
_AI_MODEL = None

def soruyu_milvusta_ara(kullanici_sorusu: str, kac_cevap_getirsin: int = 9):
    """
    Bu fonksiyon tek bir amacı yerine getirir (RAG'ın R kısmı: Retrieval):
    1. Kullanıcının metin sorusunu (Örn: "Mevlana kimdir?") anlamsal sayılara (Vektör) çevirir.
    2. Milvus Veritabanına bağlanır.
    3. Veritabanındaki binlerce parça arasından soru vektörüne en yakın/benzer (Semantic) parçaları bulur.
    4. Sadece bulunan metinleri okunaklı bir liste olarak geri döndürür.
    """
    global _AI_MODEL
    
    if _AI_MODEL is None:
        print("İlk soru sorulduğu için AI Model hafızaya yükleniyor (Lütfen bekleyin)...\n")
        _AI_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
    # ADIM 1: SORUYU VEKTÖRE (SAYILARA) ÇEVİRME
    # DİKKAT ÖĞRETİCİ NOT (Neden [0].tolist() yapıyoruz?):
    # .encode() işlemi aslen liste tabanlı çalışır. [kullanici_sorusu] diyerek tek elemanlı bir liste verdik.
    # Model cevabı geri verirken dışarıda bir liste kabuğuyla verir: [ [0.12, 0.45, -0.67] ]
    # Veritabanı sadece içerdeki saf vektörü [0.12, 0.45...] istediği için, [0] diyerek o dış kabuğu kırıyoruz.
    soru_vektoru = _AI_MODEL.encode([f"query: {kullanici_sorusu}"])[0].tolist()
    
    # ---------------------------------------------------------
    # ADIM 2: VERİTABANINDA (MILVUS) ANLAMSAL ARAMA YAPMA
    # ---------------------------------------------------------
    client = connectClient()
    
    # data: Sorgulanacak vektör listemiz (Şu an tek sorumuz olduğu için [soru_vektoru] olarak veriyoruz)
    # output_fields: Eşleşmelerin hangi kısımlarını istiyoruz? Bize orijinal "text" (metin) kısımları lazım.
    arama_sonuclari = client.search(
        collection_name=COLLECTION_NAME,
        data=[soru_vektoru], 
        limit=kac_cevap_getirsin,
        output_fields=["text"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
    )
    
    # ---------------------------------------------------------
    # ADIM 3: KARMAŞIK SONUÇLARI TEMİZLEME
    # ---------------------------------------------------------
    # Milvus bize bir sürü gereksiz alt veri (ID'ler, uzaklık skorları vb.) yollar.
    # Biz aralarından sadece asıl "text" leri cımbızlayıp temiz bir listeye dizeceğiz.
    
    bulunan_temiz_metinler = []
    
    # arama_sonuclari[0] -> Çünkü Milvus'a sadece 1 soru sorduk, o yüzden ilk listenin içindeki eşleşmeleri dönüyoruz.
    for eslesme in arama_sonuclari[0]:
        # 'entity' demek Milvus'un içindeki bizim kaydettiğimiz orijinal sözlük (dictionary) verimizdir.
        asit_metin = eslesme["entity"]["text"]
        bulunan_temiz_metinler.append(asit_metin)
        
    return bulunan_temiz_metinler


if __name__ == "__main__":
    # Kodun çalıştığını test etmek için örnek bir senaryo
    ornek_soru = "Konya hangi göller bölgesine yakındır?"
    
    print(f"Soru: {ornek_soru}\n")
    print("Milvus veritabanında taranıyor...\n")
    
    try:
        cevap_parcalari = soruyu_milvusta_ara(ornek_soru, kac_cevap_getirsin=9)
        print("--- BULUNAN EN YAKIN METİNLER ---")
        for i, metin in enumerate(cevap_parcalari, 1):
            print(f"{i}. Parça:\n{metin}\n")
    except Exception as e:
        print(f"HATA OLUŞTU: {e}")
        print("Not: Milvus'un çalıştığından emin misin?")
