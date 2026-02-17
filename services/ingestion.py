import os
from typing import List
from pypdf import PdfReader # pymupdf yoktu, pypdf kullanıyoruz
from sentence_transformers import SentenceTransformer
from core.database import connectClient
from config.settings import COLLECTION_NAME, EMBEDDING_MODEL_NAME

# 1. ADIM: PDF DOSYASINI OKUMA
def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        # Sayfadaki yazıyı oku ve ana metne ekle
        text += page.extract_text() + "\n"
    return text

# 2. ADIM: METNİ PARÇALARA BÖLME (CHUNKING)
def split_text(text, chunk_size=200):
    # Metni kelimelere böl (boşluklardan ayır)
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        # Eğer elimizdeki kelime sayısı sınıra ulaştıysa (örn: 200 kelime)
        if len(current_chunk) >= chunk_size:
            # Kelimeleri tekrar birleştirip cümleyi listeye ekle
            chunks.append(" ".join(current_chunk))
            current_chunk = [] # Sepeti boşalt
            
    # Eğer en sonda sepetin dibinde kalanlar varsa onları da ekle
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

# 3. ADIM: VEKTÖR OLUŞTURMA (EMBEDDING)
def get_embeddings(texts):
    # Bu model, kelimeleri sayısal haritalara (vektörlere) çevirir.
    # 'all-MiniLM-L6-v2' hızlı ve hafif bir modeldir.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts)
    return embeddings.tolist()

# 4. ADIM: HER ŞEYİ BİRLEŞTİRME VE KAYDETME
def ingest_data(pdf_folder):
    # Veritabanına bağlan (Merkezi fonksiyonu kullanıyoruz)
    client = connectClient()
    
    # Klasördeki her dosya için:
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"İşleniyor: {filename}")
            file_path = os.path.join(pdf_folder, filename)
            
            # A. Oku
            full_text = load_pdf(file_path)
            
            # B. Böl
            chunks = split_text(full_text)
            print(f"{len(chunks)} parçaya bölündü.")
            
            # C. Vektörle
            vectors = get_embeddings(chunks)
            
            # D. Kaydet
            data_to_insert = []
            for i, chunk in enumerate(chunks):
                data_to_insert.append({
                    "text": chunk,
                    "vector": vectors[i]
                })
                
            client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
            print(f"{filename} veritabanına kaydedildi!")

if __name__ == "__main__":
    # 'data' klasörünü bul ve işlemi başlat
    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    if os.path.exists(data_folder):
        ingest_data(data_folder)
    else:
        print("Data klasörü yok!")
