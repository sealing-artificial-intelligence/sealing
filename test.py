from sealing import SEAL
from sentence_transformers import SentenceTransformer

seal = SEAL(api_key="YOUR_API_KEY")

vectors = seal.pull_vectors()
if vectors is not None:
    print("Retrieved compressed vectors:", vectors)