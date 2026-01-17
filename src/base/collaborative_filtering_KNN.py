
"""
Sviluppo del sistema Collaborative Filtering con KNN.

Obiettivi:
  1) Caricare e filtrare il dataset (min review per user/item + sample)
  2) Eseguire un grid search per trovare gli iperparametri migliori di KNNBasic
  3) Valutare la configurazione ottimale con split diversi (RMSE/MSE/MAE)
  4) Allenare su full trainset e costruire una prediction matrix (users x items)
  5) Salvare la prediction matrix su disco per riuso (clustering + top-N)
  6) Clustering con K-Means

Output:
  - stampe a terminale
  - grafici per elbow method e silhouette nel clustering

Nota:
  - Questo script è pensato come entry-point “run and print”.
"""

from imports import *


# Config.

DATASET_DIR = Path("C:/Users/gcaso/Documents/progetti_uni/RS-MIGA/data/complete_files/")
OUTPUT_DIR = Path("C:/Users/gcaso/Documents/progetti_uni/RS-MIGA/data/products/")
REVIEWS_PARQUET = "Books.parquet"
REC_MATRIX_NAME = "KNN_rec_matrix.npy"
_N = 60_000  # sample size
_X = 20  # minimo reviews per user e per item
RANDOM_STATE = 42


# Griglia parametri per grid search.

PARAM_GRID_KNN = {
    'k': [3, 5, 10, 15, 20, 25],
    'sim_options': {
        'name': ['cosine', 'pearson', 'msd'],
        'user_based': [True, False],
        'min_support': [1, 2, 3, 5],
    },
}

pd.set_option('display.float_format', '{:.5f}'.format)


'''
Funzioni helpers.

'''
def _filter_and_sample_reviews(reviews_full: pd.DataFrame,
                               x: int,
                               n: int,
                               seed: int) -> pd.DataFrame:
    '''
    Filtra le reviews e ne fa un sample.

    Args: 
        - reviews_full: il dataframe completo delle reviews,
        - x: il numero medio di reviews per user/item,
        - n: il numero di righe da samplare,
        - seed: il seed da passare al parametro 'random_state' della sample().
    
    Returns:
        - dataframe delle reviews filtrate e samplate.
    '''
    reviews_per_user = reviews_full.groupby('user_id').size()
    reviews_per_item = reviews_full.groupby('parent_asin').size()
    user_validi = reviews_per_user[reviews_per_user > x].index
    item_validi = reviews_per_item[reviews_per_item > x].index
    reviews_filtered = reviews_full[
        reviews_full['user_id'].isin(user_validi) &
        reviews_full['parent_asin'].isin(item_validi)
    ].copy()

    if n is not None and len(reviews_filtered) > n:
        reviews_filtered = reviews_filtered.sample(n, random_state=seed).copy()

    return reviews_filtered


def _build_pred_matrix(algo,
                       users: np.ndarray,
                       items: np.ndarray) -> np.ndarray:
    '''
    Matrice (U x I) con estimated rating per ogni user-item (solo su subset
    users/items).

    Args:
        - algo: l'algoritmo da usare per le prediction dei valori,
        - users: array degli 'u_id' degli utenti,
        - items: array dei 'parent_asin' dei prodotti.
    
    Returns: 
        - L'array bidimensionale dei rating predetti per ogni coppia
          ('u_id', 'parent_asin')
    '''
    U = len(users)
    I = len(items)
    mat = np.empty((U, I), dtype=np.float32)
    for ui, uid in enumerate(users):
        if ui % 100 == 0:
            print(f'  Predizioni user {ui}/{U}')
        row = []
        for iid in items:
            row.append(algo.predict(str(uid), str(iid)).est)
        mat[ui, :] = np.asarray(row, dtype=np.float32)
    return mat


def _topn_from_matrix(pred_matrix: np.ndarray,
                      users: np.ndarray,
                      items: np.ndarray,
                      n: int = 10) -> pd.DataFrame:
    
    '''
    Restituisce un DF user_id -> lista ordinata di item consigliati (solo pred).

    Args:
        - pred_matrix: l'array bidimensionale dei rating predetti per ogni
          coppia ('u_id', 'parent_asin'),
        - users: array degli 'u_id' degli utenti,
        - items: array dei 'parent_asin' dei prodotti,
        - n: le top n raccomandazioni che vogliamo mostrare.
    
    Returns: 
        - il dataframe con una riga per ogni 'u_id' e n colonne per le n
          raccomandazioni da mostrare.
    '''
    def sort_columns(row):
        sorted_cols = sorted(zip(items, row), key=lambda x: x[1], reverse=True)
        return [c[0] for c in sorted_cols[:n]]

    res_df = pd.DataFrame(pred_matrix, index=users, columns=items)
    rec_lists = pd.DataFrame(list(res_df.apply(sort_columns, axis=1)), 
                             index=res_df.index)
    return rec_lists


# Caricamento e preprocessing delle reviews dal file completo.

reviews_full = pd.read_parquet(DATASET_DIR / REVIEWS_PARQUET)
need = {'user_id', 'parent_asin', 'rating'}
missing = need - set(reviews_full.columns)
if missing:
    raise ValueError(f'Colonne mancanti: {missing}')

reviews_full['rating'] = pd.to_numeric(reviews_full['rating'], errors='coerce')
reviews_full = (reviews_full.dropna(
    subset=['user_id', 'parent_asin', 'rating']).copy())

reviews_final = _filter_and_sample_reviews(reviews_full,
                                           x=_X,
                                           n=_N,
                                           seed=RANDOM_STATE)

print('\n\n### REVIEWS FINALI (filtrate + sample)\n')
print(reviews_final.shape)
print((f'Users: {reviews_final['user_id'].nunique()} '
       f'| Items: {reviews_final['parent_asin'].nunique()}'))


# Salvataggio reviews filtrate.

out_path = DATASET_DIR / "Books_filtered.parquet"
reviews_final.to_parquet(out_path, index=False)


# Caricamento delle reviews già filtrate e creazione del dataset Surprise.

reviews_final = pd.read_parquet(DATASET_DIR / 'Books_filtered.parquet')
reader = Reader(rating_scale=(1, 5))
dataset_knn = Dataset.load_from_df(
    reviews_final[['user_id', 'parent_asin', 'rating']],
    reader)


# Test KNN baseline (parametri standard).

sim_options_default = {'name': 'cosine', 'user_based': True, 'min_support': 2}
knn_default = KNNBasic(k=10, sim_options=sim_options_default)
start_time = time.time()
cv_results = cross_validate(knn_default,
                            dataset_knn,
                            measures=['RMSE', 'MSE'],
                            cv=3,
                            verbose=True)
end_time = time.time()

print('\n### TEST KNN CON PARAMETRI DEFAULT:\n')
print('k=10, name=cosine, user_based=True, min_support=2')
print(f'RMSE medio: {cv_results['test_rmse'].mean():.4f}'
      f' +- {cv_results['test_rmse'].std():.4f}')
print(f'MSE  medio: {cv_results['test_mse'].mean():.4f}'
      f' +- {cv_results['test_mse'].std():.4f}')
print(f'Tempo: {end_time - start_time:.2f} secondi')


'''
GridSearch 'dettagliato' per trovare gli iperparametri migliori.

Viene usata la PARAM_GRID_KNN per vagliare ogni possibile combinazione
(sensata) di parametri, viene poi eseguito il GridSearch e vengono stampati i
risultati migliori; viene creato anche un dataframe che contiene tutti i
risultati e viene ordinato per RMSE crescente. Sono stati usati come misure RMSE
e MSE.
'''

print('\n\n### GRID SEARCH KNN\n')

gs_knn = GridSearchCV(KNNBasic,
                      PARAM_GRID_KNN,
                      measures=['rmse', 'mse'],
                      cv=3,
                      n_jobs=1)

start_time = time.time()
gs_knn.fit(dataset_knn)
end_time = time.time()

print(f'Miglior RMSE: {gs_knn.best_score['rmse']:.6f}')
print(f'Miglior MSE : {gs_knn.best_score['mse']:.6f}')
print(f'Migliori parametri (RMSE): {gs_knn.best_params['rmse']}')
print(f'Tempo Grid Search: {end_time - start_time:.2f} secondi')

cv_results = gs_knn.cv_results
results_df = pd.DataFrame({
    'k':
        [p['k'] for p in cv_results['params']],
    'similarity':
        [p['sim_options']['name'] for p in cv_results['params']],
    'user_based':
        [p['sim_options']['user_based'] for p in cv_results['params']],
    'min_support':
        [p['sim_options']['min_support'] for p in cv_results['params']],
    'mean_rmse':
        cv_results['mean_test_rmse'],
    'std_rmse':
        cv_results['std_test_rmse'],
    'mean_mse':
        cv_results['mean_test_mse'],
    'std_mse':
        cv_results['std_test_mse'],
    'rank_rmse':
        cv_results['rank_test_rmse'],
})

results_df_sorted = results_df.sort_values('mean_rmse')

print('\n\n### TUTTE LE COMBINAZIONI ORDINATE PER RMSE:')
for _, row in results_df_sorted.iterrows():
    user_item = 'User' if row['user_based'] else 'Item'
    print(
        f'Rank {int(row['rank_rmse']):2d} | '
        f'k={int(row['k']):2d} | sim={row['similarity']:7s} '
        f'| {user_item}-based | '
        f'min_sup={int(row['min_support']):1d} | '
        f'RMSE={row['mean_rmse']:.4f}+-{row['std_rmse']:.4f} | '
        f'MSE={row['mean_mse']:.4f}+-{row['std_mse']:.4f}'
    )

print('\n\n### TOP 10 CONFIGURAZIONI:')
print(results_df_sorted.head(10))


'''
Test finale con i parametri migliori.

Non viene usata best_params ma vengono usati dei best_params 'artificiali'. 
Nel report viene spiegato perché.

Best_params:
    - k=20
    - sim=cosine
    - User-based
    - min_sup=3
'''

#best_params = gs_knn.best_params["rmse"]
#sim_options_best = best_params["sim_options"]
#k_best = best_params["k"]
sim_options_best = {
    'name' : 'cosine',
    'similarity' : 'cosine',
    'user_based' : True

}
k_best = 20

print('\n\n### TEST FINALE KNN (best RMSE)\n')
print(f'best k={k_best}, sim={sim_options_best}')

for test_size, seed in [(0.1, 42), (0.2, 19), (0.3, 27)]:
    trainset, testset = train_test_split(dataset_knn,
                                         test_size=test_size,
                                         random_state=seed)
    algo = KNNBasic(k=k_best, sim_options=sim_options_best)
    algo.fit(trainset)
    preds = algo.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)
    mse = accuracy.mse(preds, verbose=False)
    print(f'test_size={test_size:.1f}, seed={seed}: RMSE={rmse:.6f} '
          f'| MSE={mse:.6f}')


'''
Costruzione della matrice di predizione.

Viene innanzitutto fatto il train dell'algoritmo sul trainset totale, viene poi
riempita tutta la matrice di rating.
La matrice di rating viene salvata per non doverla ricalcolare ogni volta
durante il clustering.
'''

print('\n\n### COSTRUZIONE MATRICE DI PREDIZIONE (KNN)\n')

trainset_full = dataset_knn.build_full_trainset()
algo_full = KNNBasic(k=k_best, sim_options=sim_options_best)
algo_full.fit(trainset_full)
users = reviews_final['user_id'].unique().astype(str)
items = reviews_final['parent_asin'].unique().astype(str)

start_time = time.time()
pred_matrix = _build_pred_matrix(algo_full, users, items)
end_time = time.time()

np.save(OUTPUT_DIR / REC_MATRIX_NAME, pred_matrix)
print(f'\nMatrice salvata: {OUTPUT_DIR / REC_MATRIX_NAME} '
      f'| shape={pred_matrix.shape}')
print(f'Tempo impiegato per calcolare la matrice: '
      f'{(end_time - start_time) / 60:2f} minuti')


'''
Clustering con K-Means e analisi dei cluster.

Innanzitutto viene valutato il numero di k ottimale tramite Inertia (per
applicare l'elbow method) e Silhouette Score.
Viene poi applicato K-Means con il k migliore e viene fatta l'analisi dei
cluster con PCA.
'''
print('\n\n### CLUSTERING KMEANS\n')

##### DA CANCELLARE NEL FILE FINALE
#users = reviews_final['user_id'].unique().astype(str)
#items = reviews_final['parent_asin'].unique().astype(str)
#pred_matrix = np.load(f'{OUTPUT_DIR}/KNN_rec_matrix.npy')
#####

user_profiles = pred_matrix
scaler = RobustScaler()
user_profiles_scaled = scaler.fit_transform(user_profiles)

Ks = range(2, 11, 1)
inertias = []
silhouette_scores = []

for k in Ks:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(user_profiles_scaled)
    inertias.append(kmeans.inertia_)
    sil = silhouette_score(user_profiles_scaled, kmeans.labels_)
    silhouette_scores.append(sil)
    print(f'K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil:.3f}')

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(list(Ks), inertias, 'bo-')
plt.xlabel('Numero di cluster (k)')
plt.ylabel('Inertia')
plt.title('Inertia - Metodo del Gomito')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(list(Ks), silhouette_scores, 'ro-')
plt.xlabel('Numero di cluster (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.grid(True)
plt.tight_layout()
plt.show()

#optimal_k = int(Ks[int(np.argmax(silhouette_scores))])
optimal_k = 4
print(f'\n\n### Applicazione K-means con k={optimal_k}')

kmeans_final = KMeans(n_clusters=optimal_k,
                      random_state=RANDOM_STATE,
                      n_init=10)
user_clusters = kmeans_final.fit_predict(user_profiles_scaled)

print('\n\n### Analisi cluster:')
unique, counts = np.unique(user_clusters, return_counts=True)
cluster_stats = []
user_ids = reviews_final['user_id'].unique()
for cluster_id in range(optimal_k):
    cluster_users_mask = user_clusters == cluster_id
    cluster_users = [user_ids[i] for i in range(len(user_ids))
                     if cluster_users_mask[i]]
    cluster_data = reviews_final[reviews_final['user_id'].isin(cluster_users)]
    stats = {
        'cluster_id': cluster_id,
        'n_users': len(cluster_users),
        'avg_rating': cluster_data['rating'].mean(),
        'std_rating': cluster_data['rating'].std(),
        'avg_reviews_per_user': len(cluster_data) / len(cluster_users),
        'total_reviews': len(cluster_data)
    }
    cluster_stats.append(stats)
    print(f'\n\nCluster {cluster_id}:')
    print(f'\t- Utenti: {stats["n_users"]}')
    print(f'\t- Rating medio: {stats["avg_rating"]:.3f}')
    print(f'\t- Deviazione std rating: {stats["std_rating"]:.3f}')
    print(f'\t- Recensioni medie per utente: '
          f'{stats["avg_reviews_per_user"]:.1f}')

pca = PCA(n_components=2, random_state=42)
user_profiles_pca = pca.fit_transform(user_profiles_scaled)

plt.figure(figsize=(6, 10))
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray',
          'olive', 'cyan']
for cluster_id in range(optimal_k):
    cluster_points = user_profiles_pca[user_clusters == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               c=colors[cluster_id], label=f'Cluster {cluster_id}', alpha=0.6)
    
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
plt.title('Cluster degli Utenti (PCA)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 5))
for i, cluster_id in enumerate(range(optimal_k)):
    fig.add_subplot(1, 5, i+1)
    cluster_users_mask = user_clusters == cluster_id
    cluster_users = [user_ids[j] for j in range(len(user_ids))
                     if cluster_users_mask[j]]
    cluster_data = reviews_final[reviews_final['user_id'].isin(cluster_users)]
    sns.countplot(data=cluster_data, x='rating', palette='rocket')
    plt.title(f'Cluster {cluster_id}: '
              f'Distribuzione Rating\n({len(cluster_users)} utenti)')
    plt.xlabel('Rating')
    plt.ylabel('Frequenza')
plt.tight_layout()    
plt.show()

plt.figure(figsize=(6, 6))
plt.subplot(2, 3, 6)
cluster_means = [stats['avg_rating'] for stats in cluster_stats]
cluster_ids = [stats['cluster_id'] for stats in cluster_stats]
plt.bar(cluster_ids, cluster_means, color='skyblue')
plt.xlabel('Cluster ID')
plt.ylabel('Rating Medio')
plt.title('Rating Medio per Cluster')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

centroids = kmeans_final.cluster_centers_
cosine_sim_matrix = cosine_similarity(centroids)
plt.figure(figsize=(8, 6))
sns.heatmap(cosine_sim_matrix, 
           annot=True, 
           cmap='rocket', 
           center=0,
           xticklabels=[f'Cluster {i}' for i in range(optimal_k)],
           yticklabels=[f'Cluster {i}' for i in range(optimal_k)])
plt.title('Similarità Coseno tra Centroidi dei Cluster')
plt.tight_layout()
plt.show() 


# Top N recommendations.

print('\n\n### TOP-N RECOMMENDATIONS\n')
rec_lists = _topn_from_matrix(pred_matrix, users, items, n=10)
print(rec_lists.head())

