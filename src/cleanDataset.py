import os

import pandas as pd

# Obtener ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construir ruta al archivo
file_path = os.path.join(BASE_DIR, "data", "raw", "train.csv")

df = pd.read_csv(file_path)

output_path = os.path.join(BASE_DIR, "data", "processed", "clean_data_v1.csv")
df.to_csv(output_path, index=False)
