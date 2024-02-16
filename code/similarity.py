import pandas as pd
import numpy as np

class Similarity:
    def cosine_similarity(df: pd.DataFrame) -> float:
        """
        Calculate Cosine Similarity

        Returns:
            float: The cosine similarity between the two arrays.
        """
        arr = df.T.values 
        # print(f"shape of arr: {arr.shape}")
        norms = np.linalg.norm(arr, axis=1)
        norm_arr = arr / norms[:, np.newaxis] 
        cos_sim = np.dot(norm_arr, norm_arr.T)
        np.fill_diagonal(cos_sim, 0)
        return cos_sim
