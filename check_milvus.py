from pymilvus import connections, utility, Collection

from config.settings import MILVUS_URI, MILVUS_TOKEN

def milvus_durumunu_goster():
    # Projedeki gerçek Milvus bilgilerimizle bağlanalım:
    try:
        connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)
        print("Milvus'a bağlanıldı! İçerideki veriler taranıyor...\n")
        
        tablolar = utility.list_collections()
        
        if not tablolar:
            print("❌ Milvus içinde şu an HİÇBİR tablo veya veri yok. Tamamen boş/temiz.")
            return

        print("📊 MİLVUS'TA BULUNAN TABLOLAR (COLLECTIONS):")
        print("-" * 50)
        
        for tablo_adi in tablolar:
            try:
                koleksiyon = Collection(tablo_adi)
                resmi_kayit_sayisi = koleksiyon.num_entities
                print(f"🔹 Tablo Adı: {tablo_adi}")
                print(f"🔸 İçindeki Metin Parçası (Vektör) Sayısı: {resmi_kayit_sayisi} adet")
                print("-" * 50)
            except Exception as e:
                print(f"🔹 Tablo Adı: {tablo_adi} (Okunurken hata oluştu: {e})")
                
    except Exception as e:
        print(f"Milvus'a bağlanılamadı. Docker üzerinde Milvus'un çalıştığından emin ol.\nHata: {e}")

if __name__ == "__main__":
    milvus_durumunu_goster()
