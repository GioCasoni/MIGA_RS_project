
"""
Analisi esplorativa (EDA) del dataset Reviews per la categoria Books.

Include:
  - statistiche descrittive dei rating
  - sparsity (approssimata) user-item
  - distribuzioni: rating, #reviews per user, #reviews per item (scala log)
  - correlazioni (Pearson/Kendall/Spearman) su colonne numeriche disponibili:
      rating, helpful_vote, verified_purchase, images (-> has_image/num_images)

Output:
  - stampe a terminale
  - grafici (matplotlib / seaborn)

Note:
  Lo script campiona una porzione del dataset per velocità (SAMPLE_N).

"""

from imports import *

# Configurazione parametri e caricamento.


DATASET_DIR = Path("C:/Users/gcaso/Documents/progetti_uni/RS-MIGA/data/complete_files")
REVIEWS_PARQUET = "Books.parquet" 
SAMPLE_N = 50_000  
RANDOM_STATE = 42

pd.set_option("display.float_format", "{:.5f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

reviews = pd.read_parquet(DATASET_DIR / REVIEWS_PARQUET)

required_cols = {"user_id", "parent_asin", "rating"}
missing = required_cols - set(reviews.columns)
if missing:
    raise ValueError(f"Colonne mancanti nel parquet reviews: {missing}")

if SAMPLE_N is not None:
    reviews = reviews.sample(SAMPLE_N, random_state=RANDOM_STATE).copy()

print("\n\n### INFO DATASET REVIEWS (Books)\n")
print(f'Sample size: {SAMPLE_N}')
print(f'Numero di righe: {reviews.shape[0]}')
print(f'Numero di colonne: {reviews.shape[1]}')
print(f'Tipi di dato delle colonne:\n{reviews.dtypes}')
print(f'Numero di valori nulli:\n{reviews.isnull().sum()}')

reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")


# Statistiche di base sui rating.

print("\n\n### STATISTICHE DESCRITTIVE RATING\n")
print(reviews["rating"].describe())

n_users = reviews["user_id"].nunique()
n_items = reviews["parent_asin"].nunique()
print(f"\nUtenti unici: {n_users}")
print(f"Prodotti unici (parent_asin): {n_items}")

n_ratings = len(reviews)
sparsity = 1.0 - (n_ratings / (n_users * n_items))
print(f"Sparsità (approx): {sparsity:.6f}")


# Grafico: distribuzione rating.

plt.figure(figsize=(10, 5))
sns.histplot(reviews["rating"].dropna(), bins=10, kde=False, palette='rocket')
plt.title("Distribuzione Rating")
plt.xlabel("Rating")
plt.ylabel("Frequenza")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()



# Analisi distribuzione recesioni per user/item e grafico.

reviews_per_user = reviews.groupby("user_id").size()
reviews_per_item = reviews.groupby("parent_asin").size()

print("\nRecensioni medie per utente:", reviews_per_user.mean())
print("Recensioni medie per prodotto:", reviews_per_item.mean())

value_counts_users = Counter(reviews_per_user.values)
values_users = list(value_counts_users.keys())
counts_users = list(value_counts_users.values())

plt.figure(figsize=(12, 6))
sns.barplot(x=values_users,
            y=counts_users,
            hue=values_users,
            palette="rocket",
            legend=False)
plt.xlabel("Numero di recensioni")
plt.ylabel("Numero di utenti")
plt.yscale("log")
plt.title("Distribuzione utenti per numero di recensioni (scala log)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

value_counts_items = Counter(reviews_per_item.values)
values_items = list(value_counts_items.keys())
counts_items = list(value_counts_items.values())

plt.figure(figsize=(12, 6))
sns.barplot(x=values_items,
            y=counts_items,
            hue=values_items,
            palette="rocket",
            legend=False)
plt.xlabel("Numero di recensioni")
plt.ylabel("Numero di prodotti")
plt.yscale("log")
plt.title("Distribuzione prodotti per numero di recensioni (scala log)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


'''
Analisi della correlazione sulle colonne disponibili

In primis cerca la correlazione tra numero di voti positivi e presenza/numero
di immagini, poi crea uno scatterplot per la correlazione numero di immagini vs 
numero di voti positivi, poi crea tre heatmap, per visualizzare la correlazione
tra varie colonne, una per ogni misura di similarità.
'''

print("\n\n### ANALISI CORRELAZIONE\n")

reviews_corr = reviews.copy()

def count_images(x):
        return len(x.tolist())

def has_images(x):
    if count_images(x) >= 1:
        return True
    else:
        return False

pd.set_option('display.max_colwidth', None)

reviews_corr['has_image'] = reviews_corr['images'].apply(has_images)
reviews_corr['num_images'] = reviews_corr['images'].apply(count_images)
#print(reviews_corr['has_image'].value_counts())

mean_votes_w_images = reviews_corr.groupby('has_image')['helpful_vote'].mean()
print(f'Voti medi reviews con e senza immagini: {mean_votes_w_images}')
print('Correlazione tra presenza di immagini e numero di voti helpful\n' +
      f'{reviews_corr[['has_image', 'helpful_vote']].corr()}')
print('Correlazione tra numero di immagini e numero di voti helpful:\n' +
      f'{reviews_corr[['num_images', 'helpful_vote']].corr()}')

sns.scatterplot(x='num_images', y='helpful_vote', data=reviews_corr)
plt.title("Numero immagini vs Voti utili")
plt.xlabel("Numero di immagini nella recensione")
plt.ylabel("Helpful Votes")
plt.ylim(0, reviews_corr['helpful_vote'].quantile(0.99))
plt.show()

reviews_corr['has_image'] = reviews_corr['has_image'].astype(int)
reviews_corr['verified_purchase'] = (reviews_corr['verified_purchase']
                                     .astype(int))
corr_cols = (['rating', 'helpful_vote', 'verified_purchase',
              'has_image', 'num_images'])
corr_matrix_pearson = reviews_corr[corr_cols].corr(method='pearson')
corr_matrix_kendall = reviews_corr[corr_cols].corr(method='kendall')
corr_matrix_spearman = reviews_corr[corr_cols].corr(method='spearman')

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix_pearson, annot=True, cmap='rocket', vmin=-1, vmax=1)
plt.title("Heatmap delle correlazioni tra variabili - Pearson")
plt.tick_params(axis='both', labelsize=8, labelrotation=0.45)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix_kendall, annot=True, cmap='rocket', vmin=-1, vmax=1)
plt.title("Heatmap delle correlazioni tra variabili - Kendall")
plt.tick_params(axis='both', labelsize=8, labelrotation=0.45)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix_spearman, annot=True, cmap='rocket', vmin=-1, vmax=1)
plt.title("Heatmap delle correlazioni tra variabili - Spearman")
plt.tick_params(axis='both', labelsize=8, labelrotation=0.45)

plt.show()
