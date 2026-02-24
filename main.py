import sys
import pathlib

# =========================================================
# KONYA RAG - ANA ORKESTRA ŞEFİ (main.py)
# =========================================================
# Bu dosya bizim "Vites Kolumuzdur". Kullanıcı diğer karmaşık dosyalarla (services vb.)
# uğraşmaz, sadece bu dosyayı çalıştırır ve karşısına çıkan menüden ne yapmak
# istediğini seçer. Biz de kullanıcının seçimine göre ilgili dosyayı (modülü) yardıma çağırırız.

# 1. HAZIRLIK: Diğer odalardaki ustaları (modülleri) bu odaya (main) çağırıyoruz:
from services.ingestion import pdf_verilerini_veritabanina_yukle
from services.retrieval import soruyu_milvusta_ara
from services.generation import secilen_metinler_ile_cevap_uret
from core.database import veritabanini_sifirla

def ekrani_temizle():
    print("\n" * 5)

def ana_menu_goster():
    """
    Kullanıcıya yapabileceği işlemleri sunan basit bir metin menüsü.
    """
    ekrani_temizle()
    print("=====================================================")
    print("        KONYA RAG SİSTEMİNE HOŞ GELDİNİZ")
    print("=====================================================")
    print("[1] Yeni PDF Verisi Yükle (Ingestion)")
    print("[2] Sisteme Soru Sor (Retrieval + Generation)")
    print("[3] Veritabanını Tamamen Temizle (Sıfırla)")
    print("[0] Çıkış")
    print("=====================================================")
    
    # input() ile kullanıcının klavyeden girdiği tuşu (1, 2, 3 veya 0) yakalayıp değişkene atıyoruz
    secim = input("Lütfen yapmak istediğiniz işlemi seçin (0/1/2/3): ")
    return secim

def senaryo_veri_yukleme():
    """
    Menüden 1 basıldığında çalışacak olan 'Hafızaya Alma' fabrikası
    """
    print("\n--- PDF VERİ YÜKLEME MODU ---")
    print("Lütfen bekleyin, 'data' klasöründeki dosyalar okunuyor...")
    
    # Data klasörünün yolunu buluyoruz (main.py'nin bir altındaki data klasörü)
    data_klasoru = pathlib.Path(__file__).parent / "data"
    
    if data_klasoru.exists():
        # Ustamızı çağırıp işi ona devrediyoruz
        pdf_verilerini_veritabanina_yukle(str(data_klasoru))
        print("Tıklama: Tüm pdf'ler başarıyla Milvus'a gömüldü!")
    else:
        print(f"HATA: {data_klasoru} isminde bir klasör bulunamadı. Lütfen klasörü oluşturun.")
        
    input("\nAna menüye dönmek için Enter'a basın...")

def senaryo_soru_sorma():
    """
    Menüden 2 basıldığında çalışacak olan 'Soru Sorma ve Cevaplama' fabrikası
    """
    print("\n--- 🤖 YAPAY ZEKA SOHBET MODU ---")
    print("Çıkmak için sorunuza 'q' veya 'çıkış' yazabilirsiniz.\n")
    
    # Kullanıcı q yazana kadar sürekli soru sorabilsin diye sonsuz döngü (while) açıyoruz
    while True:
        kullanici_sorusu = input("Sorunuz: ")
        
        # Çıkış kontrolü
        if kullanici_sorusu.lower() in ["q", "çıkış", "cikis"]:
            print("Sohbetten çıkılıyor...")
            break
            
        print("⏳ Hafıza (Milvus) taranıyor...")
        
        # 1. ADIM: Ustamızı çağırıp "Bu soruya en yakın PDF parçasını bana getir" diyoruz (Retrieval)
        bulunan_parcalar = soruyu_milvusta_ara(kullanici_sorusu, kac_cevap_getirsin=9)
        
        if bulunan_parcalar:
             print(f"✅ Hafızada ({len(bulunan_parcalar)}) adet ilgili metin bulundu.")
             print("⏳ Şimdi Yapay Zeka (Ollama) cümleyi toparlıyor...\n")
             
             # 2. ADIM: Bulduğumuz o metinleri ve soruyu diğer ustamıza (Generation) verip Türkçe cevap istiyoruz
             nihai_cevap = secilen_metinler_ile_cevap_uret(kullanici_sorusu, bulunan_parcalar)
             
             print("🤖 OLLAMA CEVABI:")
             print("------------------------------------------------")
             print(nihai_cevap)
             print("------------------------------------------------\n")
        else:
             print("❌ Özür dilerim, hafızada bu soruya uyan hiçbir kitap/pdf parçası bulamadım.")
             print("İpucu: Belki de henüz PDF yüklemediniz? (Menüden 1. seçeneği deneyin)\n")

def senaryo_veritabani_sifirla():
    """
    Menüden 3 basıldığında çalışacak olan 'Hafıza Silme' fabrikası
    """
    print("\n--- 🗑️ VERİTABANI (MİLVUS) SIFIRLAMA MODU ---")
    onay = input("DİKKAT: İçerideki tüm PDF hafızası silinecek! Emin misiniz? (e/h): ")
    
    if onay.lower() == 'e':
        veritabanini_sifirla()
    else:
        print("İptal edildi, hafıza korundu.")
        
    input("\nAna menüye dönmek için Enter'a basın...")

# =========================================================
# MOTORU ÇALIŞTIRAN ANA ŞALTER
# =========================================================
if __name__ == "__main__":
    # Program ilk açıldığında doğrudan bu sonsuz döngüye girip menüyü ekrana basar.
    while True:
        kullanici_secimi = ana_menu_goster()
        
        if kullanici_secimi == "1":
            senaryo_veri_yukleme()
        elif kullanici_secimi == "2":
            senaryo_soru_sorma()
        elif kullanici_secimi == "3":
            senaryo_veritabani_sifirla()
        elif kullanici_secimi == "0":
            print("\nSistem kapatılıyor. İyi günler!")
            # sys.exit() kodu programı tamamen durdurup terminalden atar.
            sys.exit(0)
        else:
            # 1, 2 veya 0 dışında bir tuşa basarsa fırça kayıyoruz :)
            print("\nHatalı seçim yaptınız! Lütfen sadece menüdeki sayıları kullanın.")
            input("Devam etmek için Enter'a basın...")
