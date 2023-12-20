import pandas as pd
from sentence_transformers import SentenceTransformer


model_name = "neuralmind/bert-base-portuguese-cased"
encoder = SentenceTransformer(model_name_or_path=model_name)

title = """
  Paraguaios vão às urnas neste domingo (30) para escolher novo presidente
"""

sentence = title

sentence_embedding = encoder.encode(sentence)
print (sentence_embedding)

