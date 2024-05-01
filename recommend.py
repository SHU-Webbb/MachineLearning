import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 장르 유사성이 높은 top_n의 2배 만큼 선정한 뒤, 가중평점이 높은 순으로 10개 tip_n만큼 추천 
def find_sim_movie(df, sorted_ind, title_name, top_n=10) :
    
    # 매개변수로 입력된 title_name의 영화정보를 읽는다.
    title_movie = df[df['title'] == title_name]
    
    # 추출한 title_name의 영화정보에서 id를 추출
    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, :(top_n * 2)]
    
     # 추출된 벡터를 1차원 array로 변경
    similar_indexes = similar_indexes.reshape(-1)

    # 기준 영화 인덱스 제외
    similar_indexes = similar_indexes[similar_indexes != title_index]
    
    return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]

def make_genre_mat(df):
    df['genres_literal'] = df['genres'].apply(lambda x : (' ').join(x))
    count_vect = CountVectorizer(min_df = 0. , ngram_range=(1,2)) # min_df - 최소 데이터빈도, ngram_range - 단어의 묶음을 1개 또는 2개로 지정.
    genre_mat = count_vect.fit_transform(df['genres_literal'])
    genre_sim = cosine_similarity(genre_mat, genre_mat)
    genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]
    
    return genre_sim_sorted_ind

# 영화 추천 메임 메소드
def movie_recommend():
    movies_df = pd.read_pickle('movies_df.pkl')
    genre_sim = make_genre_mat(movies_df)
    
    similar_movies = find_sim_movie(movies_df, genre_sim, 'The Godfather', 10)
   
    for row in similar_movies.index:
        print(similar_movies.loc[row, 'title'], ',', similar_movies.loc[row, 'vote_count'],',',
              similar_movies.loc[row, 'vote_average'], ',', similar_movies.loc[row, 'weighted_vote'])
        
    
# 실행
movie_recommend()