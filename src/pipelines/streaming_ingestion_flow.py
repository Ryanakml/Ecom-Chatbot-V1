from prefect import task, flow
from kafka import KafkaConsumer
import json

# Import or redefine the processing tasks from the batch flow
from data_ingestion_flow import chunk_documents, generate_embeddings, index_data_streaming, dvc_push_artifacts

@task
def consume_messages(topic_name: str):
    """Consumes messages from a Kafka topic and triggers processing."""
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers='localhost:29092',
        auto_offset_reset='earliest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    print(f"Listening for messages on topic: {topic_name}")
    for message in consumer:
        doc_id = message.value.get('doc_id')
        doc_content = message.value.get('content')
        if doc_content:
            print(f"Received new document: {message.value.get('doc_id')}")
            # Trigger a sub-flow to process this single document
            process_single_document_flow(doc_id, doc_content)

@flow(name="Process Single Document")
def process_single_document_flow(doc_id: str, document: str):
    """Processes a single document from the streaming source."""
    # The document is already a single string, so we treat it as a list of one
    chunks = chunk_documents([document])
    embeddings = generate_embeddings(chunks)

    path = f"artifacts/streaming/{doc_id}.txt"
    index_data_streaming(doc_id, embeddings)
    dvc_push_artifacts(path)

@flow(name="Streaming Data Ingestion")
def streaming_ingestion_flow():
    """A long-running flow to consume and process documents from Kafka."""
    consume_messages('new_documents')

if __name__ == "__main__":
    streaming_ingestion_flow()