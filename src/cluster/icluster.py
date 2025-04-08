import os
import json
import numpy as np


class ICluster:

    log_path: str
    log_filename: str = 'model_info.json'
    model_info: dict
    local_bandwidth_vectors: np.ndarray

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.__load_log_json()
        self.__extract_bandwidth_vectors()

    def __load_log_json(self) -> None:
        with open(
            os.path.join(self.log_path, self.log_filename), 'r'
        ) as file:
            self.model_info = json.load(file)

    def __extract_bandwidth_vectors(self):
        vectors = []
        for episode in self.model_info['bandwidth_optimization']:
            bw_str = episode['bandwidth']
            vector = json.loads(bw_str)
            vectors.append(vector)
        self.local_bandwidth_vectors = np.array(vectors)
