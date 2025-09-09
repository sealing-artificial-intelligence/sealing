import requests
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import pandas as pd
import sys
import itertools
import threading
import io
import time
import socket
import torch
import torch.nn as nn
import torch.optim as optim

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

        def spinner():
            for c in itertools.cycle(['|', '/', '-', '\\']):
                if stop_spinner:
                    break
                sys.stdout.write(f'\rGenerating data... {c}')
                sys.stdout.flush()
                time.sleep(0.1)
            sys.stdout.write('\rDone generating data!   \n')

        stop_spinner = False
        spinner_thread = threading.Thread(target=spinner)
        spinner_thread.start()

        try:
            with open(csv_file_path, 'rb') as file:
                files = {'file': (csv_file_path, file, 'text/csv')}
                headers = {'Authorization': f'Bearer {self.api_key}'}
                response = requests.post(url, files=files, headers=headers)

            stop_spinner = True
            spinner_thread.join()

            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                print("Received and parsed synthetic data successfully.")
                return df
            else:
                print(f"Error: Received status code {response.status_code}")
                print(response.text)
                return None

        except Exception as e:
            stop_spinner = True
            spinner_thread.join()
            print(f"Failed to generate data from CSV: {e}")
            return None
    
    def upload_images(self, zip_file_path: str):
        url = f"{self.base_url}/upload-zip"
        try:
            with open(zip_file_path, 'rb') as f:
                files = {
                    'file': (zip_file_path, f, 'application/zip')
                }
                data = {
                    'api_key': self.api_key
                }
                headers = {
                    'Authorization': f'Bearer {self.api_key}'
                }

                response = requests.post(url, files=files, data=data, headers=headers)

            if response.status_code == 200:
                print("Zip uploaded successfully:", response.json())
                return response.json()
            else:
                print(f"Failed to upload zip: {response.status_code} {response.text}")
                return None
        except Exception as e:
            print(f"Exception during zip upload: {e}")
            return None
        
    def download_zip(self, file_name: str, save_path: str):
        url = f"{self.base_url}/download_zip"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "api_key": self.api_key,
            "file_name": file_name
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, stream=True)
            
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"ZIP downloaded and saved to {save_path}")
                return save_path
            else:
                print(f"Failed to download ZIP: {response.status_code} {response.text}")
                return None
        except Exception as e:
            print(f"Exception during ZIP download: {e}")
            return None


    @property
    def validated(self) -> bool:
        return self._validated


class DeepSetRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rnn_hidden_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rnn = nn.GRU(hidden_dim, rnn_hidden_dim, batch_first=True)
        self.rho = nn.Sequential(
            nn.Linear(rnn_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.rnn_hidden_dim = rnn_hidden_dim
        self.hidden_state = None

    def forward(self, x):
        h = self.phi(x)
        h_sum = h.sum(dim=0, keepdim=True).unsqueeze(0)
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(1, 1, self.rnn_hidden_dim)
        else:
            self.hidden_state = self.hidden_state.detach()
        rnn_out, self.hidden_state = self.rnn(h_sum, self.hidden_state)
        out = self.rho(rnn_out.squeeze(0))
        return out.squeeze(0)

    def reset_hidden(self):
        self.hidden_state = None


class DeepSetSocket:
    def __init__(self, sock, model, optimizer, criterion, k=10):
        self.sock = sock
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.k = k

    def send(self, message: np.ndarray):
        data = message.astype(np.float32).tobytes()
        self.sock.sendall(data)

    def recv(self, expected_dim: int):
        received = []
        for _ in range(self.k):
            chunk = self.sock.recv(expected_dim * 4)
            if not chunk:
                break
            msg = np.frombuffer(chunk, dtype=np.float32)
            received.append(msg)
        if len(received) == 0:
            return None
        received_tensor = torch.tensor(np.stack(received))
        predicted = self.model(received_tensor)
        return predicted.detach().numpy()

    def train_step(self, true_message, received):
        received_tensor = torch.tensor(received)
        true_tensor = torch.tensor(true_message)
        self.optimizer.zero_grad()
        predicted = self.model(received_tensor)
        loss = self.criterion(predicted, true_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()