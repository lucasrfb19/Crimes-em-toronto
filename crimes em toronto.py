import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tabulate import tabulate

# Carregar o dataset com tratamento de erro
try:
    df = pd.read_csv("major-crime-indicators.csv")
except FileNotFoundError:
    raise FileNotFoundError("Arquivo 'major-crime-indicators.csv' não encontrado.")
except pd.errors.EmptyDataError:
    raise ValueError("O arquivo está vazio ou corrompido.")

# Verificar se as colunas necessárias existem
required_columns = {'OFFENCE', 'MCI_CATEGORY'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Colunas necessárias não encontradas no DataFrame: {required_columns - set(df.columns)}")

# Preprocessamento de texto
offences = df['OFFENCE'].fillna("").astype(str)

# Vetorização TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
tfidf_matrix = vectorizer.fit_transform(offences)

# Consulta com a palavra "theft"
query_vec = vectorizer.transform(["theft"])

# Similaridade do cosseno
cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

# 1. Similaridade perfeita
perfect_sim_indices = np.where(cosine_similarities == 1.0)[0]
perfect_matches = df.iloc[perfect_sim_indices][['OFFENCE', 'MCI_CATEGORY']].copy()
perfect_matches['Cosine_Similarity'] = 1.0

# 2. Alta similaridade (0.2 <= sim < 1.0)
high_sim_indices = np.where((cosine_similarities < 1.0) & (cosine_similarities >= 0.2))[0]
high_sim_scores = cosine_similarities[high_sim_indices]
sorted_high_sim_indices = high_sim_indices[np.argsort(-high_sim_scores)]

top_high_sim = df.iloc[sorted_high_sim_indices][['OFFENCE', 'MCI_CATEGORY']].copy()
top_high_sim['Cosine_Similarity'] = cosine_similarities[sorted_high_sim_indices]
top_high_sim = top_high_sim.drop_duplicates(subset=['OFFENCE', 'MCI_CATEGORY']).reset_index(drop=True)

# Exibir resultados formatados
print("\nOcorrências com similaridade perfeita com 'theft':")
print(tabulate(perfect_matches.head(10), headers='keys', tablefmt='pretty'))

print("\nOcorrências com alta similaridade com 'theft' (excluindo 1.0):")
print(tabulate(top_high_sim.head(10), headers='keys', tablefmt='pretty'))