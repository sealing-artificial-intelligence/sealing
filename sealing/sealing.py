import requests
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import pandas as pd
import io
class APIKeyValidationError(Exception):
    pass

class SEAL:
    def __init__(self, api_key: str, validate_url: str = "https://yearly-notable-newt.ngrok-free.app/validate-key", n_components: int = 32, batch_size: int = 64):
        self.api_key = api_key
        self.validate_url = validate_url
        self._validated = False
        self.base_url = "https://yearly-notable-newt.ngrok-free.app"
        self.endpoint = f"{self.base_url}/push-vectors"
        self.n_components = n_components
        self.batch_size = batch_size

    def push_vectors(self, vectors: list[list[float]], n_components: int = None, batch_size: int = None):
        if not vectors or not isinstance(vectors[0], list):
            raise ValueError("Input must be a non-empty list of lists (2D)")

        vector_length = len(vectors[0])

        if n_components is None:
            ratio = 160 / 384
            n_components = max(1, int(vector_length * ratio))

        payload = {
            "vectors": vectors,
            "session_key": self.api_key,
            "n_components": n_components,
            "batch_size": batch_size or self.batch_size
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.endpoint, json=payload, headers=headers)

        if response.status_code == 200:
            print("Vectors pushed successfully:", response.json())
        else:
            print("Failed to push vectors:", response.status_code, response.text)


    def pull_vectors(self):
        payload = {
            "session_key": self.api_key
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(f"{self.base_url}/pull-vectors", json=payload, headers=headers)

        if response.status_code == 200:
            return response.json().get("vectors")
        else:
            print("Failed to pull vectors:", response.status_code, response.text)
            return None
        

    def generate_data_from_csv(self, csv_file_path):
        url = "https://yearly-notable-newt.ngrok-free.app/generate-data-from-csv"

        try:
            with open(csv_file_path, 'rb') as file:
                files = {'file': (csv_file_path, file, 'text/csv')}
                headers = {'Authorization': f'Bearer {self.api_key}'}
                response = requests.post(url, files=files, headers=headers)

            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                print("Received and parsed synthetic data successfully.")
                return df
            else:
                print(f"Error: Received status code {response.status_code}")
                print(response.text)
                return None

        except Exception as e:
            print(f"Failed to generate data from CSV: {e}")
            return None

    @property
    def validated(self) -> bool:
        return self._validated
