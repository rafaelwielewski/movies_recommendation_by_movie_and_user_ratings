# Importar Pacotes
import pandas as pd
import numpy as np

# Importar arquivo de filmes e visualizar as primeiras linhas
movies = pd.read_csv('movies.csv', low_memory = False)
movies.head()

# Importar arquivo de avaliações e visualizar as primeiras linhas
ratings = pd.read_csv('ratings.csv', low_memory = False)
ratings.head()

"""Pré Processamento dos Dados"""

# Selecionar somente colunas que serão usadas
movies = movies[['movieId', 'title']]
movies.head()

# Selecionar somente colunas que serão usadas
ratings = ratings[['userId', 'movieId', 'rating']]
ratings.head()

# Agrupar as avaliações por ID do filme e contar o número de avaliações
total_ratings = ratings.groupby('movieId')['rating'].count().reset_index()
total_ratings.columns = ['movieId', 'total_ratings']

# adicionar total ratings para movies conforme o movieId
movies = movies.merge(total_ratings, on = 'movieId', how = 'left')
movies.head()

# Remover filmes nulos do banco de dados.
movies.dropna(inplace = True)
ratings.dropna(inplace = True)

# Remover filmes com menos de 1000 avaliações
movies = movies[movies['total_ratings'] >= 1000]
movies.shape

# Remover avaliações de filmes com menos de 1000 avaliações
ratings = ratings[ratings['movieId'].isin(movies['movieId'])]
ratings.shape

# Verificar quantidade de avaliações por usuário
ratings.groupby('userId').count()

# Agrupar as avaliações por userId e contar o número de avaliações
ratings_count = ratings.groupby('userId')['rating'].count()

# Filtrar os usuários com mais de 50 avaliações
y = ratings_count[ratings_count > 50].index

print(y)

# Filtrar as avaliações dos usuários com mais de 50 avaliações
ratings = ratings[ratings['userId'].isin(y)]
ratings.shape

movies.info()

ratings.info()

# Concatenar os datasets de filmes e avaliações
ratings_and_movies = ratings.merge(movies, on='movieId')
ratings_and_movies.head()

#Verificar se ha valores nulos em ratings_and_movies
ratings_and_movies.isnull().sum()

# Descartar valores duplicados verificando userId e movieId
ratings_and_movies.drop_duplicates(subset=['userId', 'movieId'], keep='first', inplace=True)
ratings_and_movies.shape

# Remover movieId
del ratings_and_movies['movieId']
ratings_and_movies.head()

# Fazer pivot da tabela
movies_pivot = ratings_and_movies.pivot(index='title', columns='userId', values='rating')
movies_pivot.head()

# Substituir ratings nulas por zero
movies_pivot.fillna(0, inplace=True)
movies_pivot.head()

# Criar uma matriz esparsa
from scipy.sparse import csr_matrix
movies_sparse = csr_matrix(movies_pivot)

# Criar modelo
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(algorithm='brute')
model_knn.fit(movies_sparse)

"""Fazer previsões de filmes para o modelo treinado"""

# fazer previsões com base em um filme
distances, sugestions = model_knn.kneighbors(movies_pivot.filter(items = ['Toy Story (1995)'], axis=0).values.reshape(1,-1))
for i in range(0, len(sugestions)):
    print(movies_pivot.index[sugestions[i]])