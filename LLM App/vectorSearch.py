from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer
import numpy as np

model_name = "jinaai/jina-embeddings-v2-small-en"
user_question = "I just discovered the course. Can I join now?"

#stmodel = SentenceTransformer(model_name)
model = TextEmbedding(model_name=model_name)

embeddings_generator = model.embed(user_question, normalize_embeddings=True)
embeddings_list = list(embeddings_generator)

#v =  stmodel.encode(user_question, normalize_embeddings=True)
print (embeddings_list[0])
print (len(embeddings_list[0]))
print(min(embeddings_list[0]))

q=embeddings_list[0]
print(np.linalg.norm(q))
print(q.dot(q))



