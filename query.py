from qdrant_client.models import Filter
from qdrant_client.http import models
from qdrant_client import QdrantClient
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer



model_name = "neuralmind/bert-base-portuguese-cased"
encoder = SentenceTransformer(model_name_or_path=model_name)

query_text = "Donald Trump"
query_vector = encoder.encode(query_text).tolist()

client = QdrantClient(path="./qdrant_data")

client.search(
    collection_name="news-articles",
    query_vector=query_vector,
    with_payload=["newsId", "title", "topics"],
    query_filter=None
)