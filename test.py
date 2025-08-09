from sealing import SEAL
from sentence_transformers import SentenceTransformer

seal = SEAL(api_key="sk-f6a0e67910794c7084d11dddff427e83")
csv_path = 'YOUR_CSV_PATH.csv'

synthetic_df = seal.generate_data_from_csv(csv_path)
print(synthetic_df.head())
