from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


model_name = "neuralmind/bert-base-portuguese-cased"
encoder = SentenceTransformer(model_name_or_path=model_name)


df = pd.read_parquet("articles.parquet")

def generate_item_sentence(item: pd.Series, text_columns=["title"]) -> str:
    return ' '.join([item[column] for column in text_columns])

df["sentence"] = df.apply(generate_item_sentence, axis=1)
df["sentence_embedding"] = df["sentence"].apply(encoder.encode)


client = QdrantClient(path="./qdrant_data")

client.create_collection(
    collection_name = "news-articles",
    vectors_config = models.VectorParams(
        size = encoder.get_sentence_embedding_dimension(),
        distance = models.Distance.COSINE,
    ),
)


metadata_columns = df.drop(["newsId", "sentence", "sentence_embedding"], axis=1).columns

def create_vector_point(item:pd.Series) -> PointStruct:
    """Turn vectors into PointStruct"""
    return PointStruct(
        id = item["newsId"],
        vector = item["sentence_embedding"].tolist(),
        payload = {
            field: item[field]
            for field in metadata_columns
            if (str(item[field]) not in ['None', 'nan'])
        }
    )

points = df.apply(create_vector_point, axis=1).tolist()

CHUNK_SIZE = 500
n_chunks = np.ceil(len(points)/CHUNK_SIZE)

for i, points_chunk in enumerate(np.array_split(points, n_chunks)):
    client.upsert(
        collection_name="news-articles",
        wait=True,
        points=points_chunk.tolist()
    )