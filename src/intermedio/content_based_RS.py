"""
Script principale del progetto intermedio.

Composto da 5 fasi:
    1) filtro reviews in modo coerente con meta (stessi item presenti in 
       reviews),
    2) crea embeddings (TF-IDF, CBoW, Transformer) con funzioni nostre e salva
       su file,
    3) analisi embeddings,
    4) content-based item-item KNN: grid search per ogni embedding + salva
       prediction matrix.
"""

from imports import *
from functions import * 

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.float_format", "{:.5f}".format)


# config

dataset_path = 'C:/Users/gcaso/Documents/progetti_uni/RS-MIGA/data/complete_files/'
products_path = 'C:/Users/gcaso/Documents/progetti_uni/RS-MIGA/data/products/'

REVIEWS_PATH = f'{dataset_path}Books.parquet'
META_PATH = f'{dataset_path}meta_Books.parquet'  
RATINGS_FILTERED_PATH = f'{dataset_path}ratings_filtered.parquet'
META_FILTERED_PATH = f'{dataset_path}meta_filtered.parquet'
MIN_REVIEWS_PER_USER = 30
MIN_REVIEWS_PER_ITEM = 30
SAMPLE_N = 100_000  
TEXT_COLUMNS = ['title', 'description', 'author', 'categories']


# Output embeddings.

TFIDF_NPZ = f'{products_path}tfidf_embeddings.npz'
CBOW_NPZ = f'{products_path}cbow_embeddings.npz'
TRF_NPZ = f'{products_path}transformer_embeddings.npz'


# Output prediction matrices (memmap .npy).

PRED_TFIDF = f'{products_path}pred_matrix_content_tfidf.npy'
PRED_CBOW = f'{products_path}pred_matrix_content_cbow.npy'
PRED_TRF = f'{products_path}pred_matrix_content_transformer.npy'

K_CLUSTERS = [5, 10, 20, 30, 40, 60, 80]
MAX_POINTS = 20_000
SEED = 42

K_LIST = [5, 10, 15, 20, 30, 40]
TEST_SIZE = 0.2


# 1) Filtering + sampling reviews e salvataggio.

print('\n\n###############################')
print('# 1) FILTER REVIEWS + META    #')
print('###############################')
t0 = time.time()
reviews_full = pd.read_parquet(REVIEWS_PATH)
reviews_f = filter_reviews_min_counts(reviews_full,
                                      min_reviews_per_user=MIN_REVIEWS_PER_USER,
                                      min_reviews_per_item=MIN_REVIEWS_PER_ITEM)
if (SAMPLE_N is not None) and (len(reviews_f) > SAMPLE_N):
    reviews_f = reviews_f.sample(n=SAMPLE_N, random_state=42).copy()
    print(f'\nSample reviews: {SAMPLE_N:,}')
meta_full = pd.read_parquet(META_PATH)
ratings_df, meta_f = filter_reviews_and_meta(reviews_f, meta_full)
ratings_df = ratings_df[['user_id', 'parent_asin', 'rating']].copy()
ratings_df.to_parquet(f'{dataset_path}ratings_filtered.parquet', index=False)
meta_f.to_parquet(f'{dataset_path}meta_filtered.parquet', index=False)
print('\nDataset finale:')
print(f'- ratings_df: {ratings_df.shape}')
print(f'- users: {ratings_df['user_id'].nunique():,}')
print(f'- items: {ratings_df['parent_asin'].nunique():,}')
t1 = time.time()
print(f'Finito di caricare, filtrare e salvare i dataset in '
      f'{(t1-t0)/60} minuti')

#ratings_df = pd.read_parquet(RATINGS_FILTERED_PATH)
#meta_f = pd.read_parquet(META_FILTERED_PATH)


# 2) Creazione embeddings e salvataggio.

print('\n\n###############################')
print('# 2) CREATE + SAVE EMBEDDINGS #')
print('###############################')
text_df = create_text_dataframe(meta_f, text_columns=TEXT_COLUMNS)


# TF-IDF 

embeddings_data_tfidf, embedder_tfidf = create_embeddings_dataframe(
    text_df=text_df,
    embedder='tfidf',
    text_columns=TEXT_COLUMNS,
    max_features=50_000,
    min_df=2,
    max_df=0.8
)
save_embeddings_npz(TFIDF_NPZ, embeddings_data_tfidf)


# CBoW / Word2Vec

embeddings_data_cbow, embedder_cbow = create_embeddings_dataframe(
    text_df=text_df,
    embedder='cbow',
    text_columns=TEXT_COLUMNS,
    vector_size=200,
    window=5,
    min_count=2,
    epochs=20
)
save_embeddings_npz(CBOW_NPZ, embeddings_data_cbow)


# Transformer HF

try:
    embeddings_data_trf, trf_info = create_embeddings_dataframe(
        text_df=text_df,
        embedder='transformer',
        text_columns=TEXT_COLUMNS,
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        batch_size=32,
        max_length=256,
        normalize=True
    )
    save_embeddings_npz(TRF_NPZ, embeddings_data_trf)
except Exception as e:
    print(f'\n[WARN] Transformer embeddings non creati '
          f'(dipendenze mancanti o config).')
    print('Errore:', e)
    embeddings_data_trf = None

#embeddings_data_tfidf = load_embeddings_npz(f'{products_path}tfidf_embeddings.npz')
#embeddings_data_cbow = load_embeddings_npz(f'{products_path}cbow_embeddings.npz')
#embeddings_data_trf = load_embeddings_npz(f'{products_path}transformer_embeddings.npz')


# 3) Analisi Embeddings.

print('\n\n###############################')
print('# 3) ANALYZE EMBEDDINGS       #')
print('###############################')
analyze_embeddings_detailed(embeddings_data_tfidf, 'TF-IDF')
analyze_embeddings_detailed(embeddings_data_cbow, 'CBoW')
if embeddings_data_trf is not None:
    analyze_embeddings_detailed(embeddings_data_trf, 'Transformer')


# Campionamento coerente per confronto equo.

rng = np.random.default_rng(SEED)
n_items_ref = embeddings_data_tfidf['embeddings'].shape[0]
sample_size = min(MAX_POINTS, n_items_ref)
sample_idx = rng.choice(n_items_ref, size=sample_size, replace=False)
df_all = []

df_tfidf = compute_kmeans_inertia_silhouette(
    embeddings_data=embeddings_data_tfidf,
    k_list=K_CLUSTERS,
    embedding_name='TF-IDF',
    random_state=SEED,
    max_points=MAX_POINTS,
    sample_idx=sample_idx
)
df_all.append(df_tfidf)

df_cbow = compute_kmeans_inertia_silhouette(
    embeddings_data=embeddings_data_cbow,
    k_list=K_CLUSTERS,
    embedding_name='CBoW',
    random_state=SEED,
    max_points=MAX_POINTS,
    sample_idx=sample_idx
)
df_all.append(df_cbow)

if embeddings_data_trf is not None:
    df_trf = compute_kmeans_inertia_silhouette(
        embeddings_data=embeddings_data_trf,
        k_list=K_CLUSTERS,
        embedding_name='Transformer',
        random_state=SEED,
        max_points=MAX_POINTS,
        sample_idx=sample_idx
    )
    df_all.append(df_trf)

df_all = pd.concat(df_all, ignore_index=True)

plot_elbow_and_silhouette_single(df_all, 'TF-IDF')

plot_elbow_and_silhouette_single(df_all, 'CBoW')

if embeddings_data_trf is not None:
    plot_elbow_and_silhouette_single(df_all, 'Transformer')

plot_elbow_and_silhouette_comparison(df_all)


# 4) Content-based KNN, GridSearch, costruzione prediction matrix.

print('\n\n############################################')
print('# 4) CONTENT-KNN: GRIDSEARCH + PRED MATRIX #')
print('############################################')
results_all = {}

df_gs_tfidf, best_tfidf = grid_search_content_knn(
    ratings_df=ratings_df,
    embeddings_data=embeddings_data_tfidf,
    k_list=K_LIST,
    metric='cosine',
    test_size=TEST_SIZE
)
results_all['content_tfidf'] = {'rmse': float(best_tfidf['rmse']),
                                'k': float(best_tfidf['n_neighbors'])}
art_tfidf = build_item_knn_index(embeddings_data_tfidf,
                                 n_neighbors=int(best_tfidf['n_neighbors']),
                                 metric='cosine')
build_and_save_prediction_matrix_memmap(ratings_df=ratings_df,
                                        artifacts=art_tfidf,
                                        out_path_npy=PRED_TFIDF,
                                        chunk_items=500)

df_gs_cbow, best_cbow = grid_search_content_knn(
    ratings_df=ratings_df,
    embeddings_data=embeddings_data_cbow,
    k_list=K_LIST,
    metric='cosine',
    test_size=TEST_SIZE)
results_all['content_cbow'] = {'rmse': float(best_cbow['rmse']),
                               'k': float(best_cbow['n_neighbors'])}
art_cbow = build_item_knn_index(embeddings_data_cbow,
                                n_neighbors=int(best_cbow['n_neighbors']),
                                metric='cosine')
build_and_save_prediction_matrix_memmap(ratings_df=ratings_df,
                                        artifacts=art_cbow,
                                        out_path_npy=PRED_CBOW,
                                        chunk_items=500)

if embeddings_data_trf is not None:
    df_gs_trf, best_trf = grid_search_content_knn(
        ratings_df=ratings_df,
        embeddings_data=embeddings_data_trf,
        k_list=K_LIST,
        metric='cosine',
        test_size=TEST_SIZE)
    results_all['content_transformer'] = {'rmse': float(best_trf['rmse']),
                                          'k': float(best_trf['n_neighbors'])}
    art_trf = build_item_knn_index(embeddings_data_trf,
                                   n_neighbors=int(best_trf['n_neighbors']),
                                   metric='cosine')
    build_and_save_prediction_matrix_memmap(ratings_df=ratings_df,
                                            artifacts=art_trf,
                                            out_path_npy=PRED_TRF,
                                            chunk_items=500)