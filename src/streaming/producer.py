import json  # untuk mengubah data Python ke JSON
from kafka import KafkaProducer  # library untuk nge-send data ke Kafka
import time  # untuk bikin timestamp

# 1. Membuat Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:29092',  # alamat Kafka server
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  # ubah Python dict jadi JSON bytes
)

# 2. Fungsi untuk publish dokumen baru
def publish_new_document(doc_id, content):
    """
    Fungsi ini mengirim dokumen baru ke Kafka topic 'new_documents'.
    """
    # Membuat message
    message = {
        'doc_id': doc_id,          # ID dokumen
        'content': content,        # isi dokumen
        'timestamp': time.time()   # waktu publish
    }

    print(f"Publishing document: {doc_id}")

    # Mengirim ke Kafka topic 'new_documents'
    producer.send('new_documents', message)

    # Flush: memastikan semua message terkirim
    producer.flush()

# 3. Fungsi tambahan: publish banyak dokumen sekaligus
def publish_multiple_documents(docs):
    """
    docs: list of dict { 'doc_id': ..., 'content': ... }
    """
    for doc in docs:
        publish_new_document(doc['doc_id'], doc['content'])

# 4. Script utama
if __name__ == '__main__':
    # Contoh dokumen tunggal
    new_doc_content = """
    ### New Feature: Super-Efficient DataFrames

    Starting with version 3.0, pandas introduces a new, highly-efficient
    DataFrame storage format. To use it, simply call `df.to_super_efficient()`
    """
    publish_new_document('doc-12345', new_doc_content)
    
    # Contoh publish banyak dokumen sekaligus
    docs_batch = [
        {'doc_id': 'doc-12346', 'content': 'Content of doc 12346'},
        {'doc_id': 'doc-12347', 'content': 'Content of doc 12347'},
        {'doc_id': 'doc-12348', 'content': 'Content of doc 12348'}
    ]
    publish_multiple_documents(docs_batch)

    print("All documents published.")