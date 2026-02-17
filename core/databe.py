from pymilvus import MilvusClient, DataType

from config.settings import MILVUS_TOKEN, MILVUS_URI, DB_NAME, COLLECTION_NAME, VECTOR_DIM

#DB var ise kullanılacak olan 
def connectClient():
    client = MilvusClient(
        uri=MILVUS_URI,
        token=MILVUS_TOKEN,
        db_name=DB_NAME
    )

    return client


def initClient():
    admin_client = MilvusClient(
        uri=MILVUS_URI,
        token=MILVUS_TOKEN
    )

    if DB_NAME not in admin_client.list_databases():
        admin_client.create_database(DB_NAME)
    
    client = connectClient()

    if client.has_collection(COLLECTION_NAME):
        return 
    
    schema = MilvusClient.create_schema(
        auto_id=True,             # ID'leri Milvus versin (1, 2, 3...)
        enable_dynamic_field=False # Sadece bizim izin verdiğimiz alanlar olsun
    )

    schema.add_field(field_name= "id", datatype=DataType.INT64, is_primary= True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim = VECTOR_DIM)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="L2"
        )
    

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
        )

    print("sorunsuz kurulum yapıldı")
 
    
if __name__ == "__main__":
    initClient()
