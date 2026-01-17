"""
Tutte le funzioni del progetto intermedio (content-based RS).

1) filtro coerente reviews <-> metadata
2) creazione + salvataggio embeddings (tfidf, cbow/word2vec, transformer HF)
3) analisi embeddings
4) item-item KNN (content-based) + grid search
5) valutazione e confronto (anche con baseline CF se vuoi)
"""

from imports import * 

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
except Exception:
    torch = None
    F = None
    AutoTokenizer = None
    AutoModel = None

try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
except Exception:
    Word2Vec = None
    simple_preprocess = None


# 1) FILTERING: reviews <-> metadata

def filter_reviews_and_meta(reviews_df: pd.DataFrame,
                            meta_df: pd.DataFrame,
                            user_col: str = 'user_id',
                            item_col: str = 'parent_asin'
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Allinea 'reviews_df' e 'meta_df' tenendo solo gli item presenti in entrambi.

    Args:
        - reviews_df: dataFrame delle recensioni/ratings, deve contenere 
          'item_col',
        - meta_df: dataFrame dei metadata prodotti, deve contenere 'item_col',
        - user_col: nome colonna utente in 'reviews_df',
        - item_col: nome colonna item ('parent_asin') presente in entrambi i
          DataFrame.

    Returns:
        - una tupla (reviews_filtered, meta_filtered) dove:
            - meta_filtered contiene solo i prodotti presenti nelle reviews,
            - reviews_filtered contiene solo le recensioni relative ai prodotti
              rimasti in meta_filtered.

    Side Effects:
        - stampa a terminale le dimensioni prima/dopo e il numero di item in
          comune.
    '''

    if item_col not in reviews_df.columns:
        raise ValueError(f'reviews_df missing \'{item_col}\'')
    if item_col not in meta_df.columns:
        raise ValueError(f'meta_df missing \'{item_col}\'')
    items_in_reviews = set(reviews_df[item_col].dropna().unique())
    meta_f = meta_df[meta_df[item_col].isin(items_in_reviews)].copy()
    items_in_meta = set(meta_f[item_col].dropna().unique())
    reviews_f = reviews_df[reviews_df[item_col].isin(items_in_meta)].copy()
    print('\n### FILTER COERENTE REVIEWS <-> META')
    print(f'- reviews: {reviews_df.shape} -> {reviews_f.shape}')
    print(f'- meta:    {meta_df.shape} -> {meta_f.shape}')
    print(f'- items in comune: {len(items_in_meta):,}')
    return reviews_f, meta_f


def filter_reviews_min_counts(reviews_df: pd.DataFrame,
                              min_reviews_per_user: int = 10,
                              min_reviews_per_item: int = 10,
                              user_col: str = 'user_id',
                              item_col: str = 'parent_asin'
                              ) -> pd.DataFrame:
    '''
    Filtra il dataset delle reviews mantenendo solo utenti e item con almeno
    N recensioni.

    - Riduce sparseness estrema,
    - stabilizza KNN e rende la valutazione più significativa,
    - riduce anche il tempo di esecuzione per grid search e costruzione matrici.

    Args:
        - reviews_df: dataFrame con almeno le colonne 'user_col', 'item_col',
        - min_reviews_per_user: numero minimo di recensioni richieste per tenere
          un utente,
        - min_reviews_per_item: numero minimo di recensioni richieste per tenere
          un item,
        - user_col: nome colonna utente,
        - item_col: nome colonna item.

    Returns:
        - dataFrame filtrato contenente solo righe con utenti e item che
          soddisfano le soglie.

    Side Effects:
        - stampa a terminale shape prima/dopo e numerosità di utenti/item
          rimasti.
    '''

    u_counts = reviews_df.groupby(user_col).size()
    i_counts = reviews_df.groupby(item_col).size()

    good_users = set(u_counts[u_counts >= min_reviews_per_user].index)
    good_items = set(i_counts[i_counts >= min_reviews_per_item].index)

    out = reviews_df[
        (reviews_df[user_col].isin(good_users) &
        reviews_df[item_col].isin(good_items))
    ].copy()

    print('\n### FILTRO MIN COUNTS')
    print(f'- min per user={min_reviews_per_user}, '
          f'min per item={min_reviews_per_item}')
    print(f'- shape: {reviews_df.shape} -> {out.shape}')
    print(f'- users: {out[user_col].nunique():,} '
          f'| items: {out[item_col].nunique():,}')

    return out



# 2) TEXT DF + EMBEDDINGS

def _normalize_text_value(val: Any) -> str:
    ''' 
    Converte un valore eterogeneo (string/list/dict/NaN) in una stringa 'clean'.

    Args:
        - val: qualsiasi valore proveniente dal DataFrame metadata.

    Returns:
        - una stringa:
            - vuota se val è null/NaN,
            - concatenazione se lista/dict,
            - cast a stringa altrimenti.
    '''

    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ''
    if isinstance(val, dict):
        if 'name' in val and isinstance(val['name'], str):
            return val['name']
        return ' '.join(str(x) for x in val.values() if x is not None)
    if isinstance(val, list):
        return ' '.join(str(x) for x in val if x is not None)
    return str(val)


def create_text_dataframe(meta_df: pd.DataFrame,
                          text_columns: List[str],
                          id_col: str = 'parent_asin'
                          ) -> pd.DataFrame:
    '''
    Crea un DataFrame 'testuale' a partire dai metadata prodotti.

    Serve perché:
        - Centralizza la preparazione dei testi (normalizzazione campi, gestione
          liste/dict).
        - Garantisce una riga per item (drop duplicati su 'id_col').
        - Fornisce l'input standard per tutte le funzioni di embedding.

    Args:
        - meta_df: dataFrame metadata (prodotti) che contiene 'id_col' e le
          colonne testuali,
        - text_columns: Lista di nomi colonna che contribuiscono al testo (es.
          'title', 'description', ...),
        - id_col: colonna identificativa del prodotto (default: 'parent_asin').

    Returns:
        - dataFrame con colonne: [id_col] + subset di 'text_columns' presenti in
          meta_df, con valori normalizzati a stringa e duplicati rimossi per
          id_col.

    Side Effects:
        Stampa a terminale colonne finali, shape e numero di duplicati rimossi.
    '''

    cols = [id_col] + [c for c in text_columns if c in meta_df.columns]
    text_df = meta_df[cols].copy()

    for c in cols:
        if c == id_col:
            continue
        text_df[c] = text_df[c].apply(_normalize_text_value)

    before = len(text_df)
    text_df = text_df.drop_duplicates(subset=[id_col], keep='first')
    after = len(text_df)

    print('\n### TEXT_DF')
    print(f'- columns: {list(text_df.columns)}')
    print(f'- shape: {text_df.shape} (removed dup: {before-after})')

    return text_df


def _concat_text_cols(text_df: pd.DataFrame,
                      text_columns: List[str]
                      ) -> List[str]:
    '''
    Concatena le colonne testuali di ogni riga in un singolo documento testuale
    per item.

    Serve perchè
        - Molti embedder (TF-IDF, Word2Vec document-level, Transformer) lavorano
          su una stringa per item. Questa funzione rende coerente come viene
          creato il "documento" finale.

    Args:
        - text_df: dataFrame prodotto da 'create_text_dataframe',
        - text_columns: colonne da concatenare (usa solo quelle effettivamente
          presenti).

    Returns:
        - lista di stringhe (docs), una per item, ottenuta concatenando i campi
          testuali.
    '''

    docs = []
    for _, row in text_df.iterrows():
        parts = []
        for c in text_columns:
            if c in text_df.columns:
                parts.append(str(row[c]))
        docs.append(" ".join(parts).strip())
    return docs


def create_embeddings_dataframe(text_df: pd.DataFrame,
                                embedder: str,
                                text_columns: List[str],
                                id_col: str = "parent_asin",
                                **kwargs
                                ) -> Tuple[Dict[str, np.ndarray], Any]:
    '''
    Factory unica per creare embeddings di tipo diverso con una sola API.

    Args:
        - text_df: dataFrame testuale (una riga per item),
        - embedder: tipo di embedding da creare. Valori ammessi:
            - 'tfidf'
            - 'cbow'
            - 'transformer' / 'hf"'/ 'huggingface',
        - text_columns: colonne testuali da utilizzare,
        - id_col: colonna id item,
        - **kwargs: parametri specifici del metodo scelto (es. max_features per
          tfidf).

    Returns:
        - una tupla (embeddings_data, embedder_obj) dove:
            a) embeddings_data è un dict con chiavi standard:
                - 'parent_asin': array di id item
                - 'embeddings': matrice numpy (n_items, dim)
            b) embedder_obj è l'oggetto addestrato/usato:
                - TfidfVectorizer, Word2Vec model, oppure dict info HF.

    Raises:
        ValueError: se 'embedder' non è uno dei valori supportati.
    '''

    embedder = embedder.lower().strip()
    if embedder == "tfidf":
        return create_tfidf_embeddings(text_df,
                                       text_columns,
                                       id_col=id_col,
                                       **kwargs)
    if embedder == 'cbow':
        return create_cbow_embeddings(text_df,
                                      text_columns,
                                      id_col=id_col,
                                      **kwargs)
    if embedder in ('transformer', 'hf', 'huggingface', 'transformers'):
        return create_transformer_embeddings(text_df,
                                             text_columns,
                                             id_col=id_col,
                                             **kwargs)
    raise ValueError("embedder must be one of: 'tfidf', 'cbow', 'transformer'")


def create_tfidf_embeddings(text_df: pd.DataFrame,
                            text_columns: List[str],
                            id_col: str = 'parent_asin',
                            max_features: int = 50_000,
                            min_df: int = 2,
                            max_df: float = 0.8,
                            ngram_range: Tuple[int, int] = (1, 2)
                            ) -> Tuple[Dict[str, np.ndarray], TfidfVectorizer]:
    '''
    Crea embeddings TF-IDF a livello documento (un documento = un item).

    Args:
        - text_df: dataFrame testuale 1 riga per item,
        - text_columns: colonne da concatenare nel documento,
        - id_col: colonna id item,
        - max_features: numero massimo di termini nel vocabolario (dimensione
          embedding),
        - min_df: frequenza minima documentale (termini troppo rari scartati),
        - max_df: frequenza massima documentale (termini troppo comuni
          scartati),
        - ngram_range: intervallo n-gram, default=(1,2).

    Returns:
        - (embeddings_data, vectorizer) dove:
            - embeddings_data['embeddings'] ha shape (n_items, 
              vocab_size_effettivo),
            - vectorizer è il TfidfVectorizer fit sul dataset.

    Side Effects:
        - stampa shape e dimensione vocabolario a terminale.
    '''

    docs = _concat_text_cols(text_df, text_columns)
    print('\n### TF-IDF EMBEDDINGS')
    vectorizer = TfidfVectorizer(max_features=max_features,
                                 min_df=min_df,
                                 max_df=max_df,
                                 ngram_range=ngram_range,
                                 stop_words='english')
    X = vectorizer.fit_transform(docs).toarray().astype(np.float32)
    embeddings_data = {
        'parent_asin': text_df[id_col].values,
        'embeddings': X
    }

    print(f'- X shape: {X.shape}')
    print(f'- vocab size: {len(vectorizer.vocabulary_):,}')

    return embeddings_data, vectorizer


def create_cbow_embeddings(text_df: pd.DataFrame,
                           text_columns: List[str],
                           id_col: str = 'parent_asin',
                           vector_size: int = 200,
                           window: int = 5,
                           min_count: int = 2,
                           epochs: int = 20,
                           workers: int = 8,
                           seed: int = 42
                           ) -> Tuple[Dict[str, np.ndarray], Any]:
    '''
    Crea embeddings Word2Vec in modalità CBOW e produce un embedding per
    item (document embedding).

    Qui usiamo embedding documento = media dei word vectors del documento.

    Args:
        - text_df: dataFrame testuale 1 riga per item,
        - text_columns: colonne da concatenare nel documento,
        - id_col: colonna id item,
        - vector_size: dimensione dei vettori Word2Vec,
        - window: dimensione finestra contesto,
        - min_count: frequenza minima parola per essere inclusa nel vocabolario,
        - epochs: epoche di training,
        - workers: thread di training,
        - seed: seed per riproducibilità.

    Returns:
        - (embeddings_data, model) dove:
            - embeddings_data['embeddings] ha shape (n_items, vector_size),
            - model è l'istanza Word2Vec addestrata.

    Raises:
        ImportError: se 'gensim' non è installato.

    Side Effects:
        Stampa shape embeddings e dimensione vocabolario a terminale.
    '''

    if Word2Vec is None or simple_preprocess is None:
        raise ImportError(f'gensim non disponibile: installa \'gensim\' '
                          f'nel tuo env.')
    docs = _concat_text_cols(text_df, text_columns)
    tokenized = [simple_preprocess(d, deacc=True) or
                 ['empty', 'text'] for d in docs]
    print('\n### CBoW / Word2Vec EMBEDDINGS')
    model = Word2Vec(sentences=tokenized,
                     vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     sg=0,workers=workers,
                     epochs=epochs,
                     seed=seed)

    doc_vecs = []
    for toks in tokenized:
        vecs = [model.wv[t] for t in toks if t in model.wv]
        doc_vecs.append(np.mean(vecs, axis=0) if vecs
                        else np.zeros(vector_size, dtype=np.float32))
    X = np.array(doc_vecs, dtype=np.float32)
    embeddings_data = {
        'parent_asin': text_df[id_col].values,
        'embeddings': X,
    }
    print(f'- X shape: {X.shape}')
    print(f'- w2v vocab: {len(model.wv):,}')

    return embeddings_data, model


def _mean_pooling(last_hidden_state: Any, attention_mask: Any) -> Any:
    '''
    Esegue mean pooling mascherato sui token (Transformer output
    -> sentence/document embedding).

    L'output di un Transformer è per-token (seq_len, hidden_dim).
    Per ottenere un vettore unico per documento dobbiamo aggregare.

    Args:
        - last_hidden_state: Tensor (batch, seq_len, hidden_dim) dal modello
          Transformer,
        - attention_mask: Tensor (batch, seq_len) con 1 per token validi e 0
          per padding.

    Returns:
        - Tensor (batch, hidden_dim) con la media dei token validi per ogni
          esempio.

    Notes:
        Questa funzione è usata internamente da 'create_transformer_embeddings'.
    '''

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def create_transformer_embeddings(text_df: pd.DataFrame,
                                  text_columns: List[str],
                                  id_col: str = "parent_asin",
                                  model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                                  batch_size: int = 32,
                                  max_length: int = 256,
                                  device: Optional[str] = None,
                                  normalize: bool = True
                                  ) -> Tuple[Dict[str, np.ndarray],
                                             Dict[str, Any]]:
    '''
    Crea embeddings di item usando un modello Transformer HuggingFace.

    Embeddings 'deep' tendono a catturare semantica meglio di TF-IDF/Word2Vec,
    soprattutto su testi ricchi, e spesso migliorano la qualità del KNN
    content-based.

    Implementazione:
        - Tokenizzazione batched (padding + truncation)
        - Forward pass del modello
        - Mean pooling mascherato
        - (opzionale) normalizzazione L2 per usare cosine in modo più stabile

    Args:
        - text_df: dataFrame testuale 1 riga per item,
        - text_columns: colonne da concatenare nel documento,
        - id_col: colonna id item,
        - model_name: modello HF (es. sentence-transformers/all-MiniLM-L6-v2),
        - batch_size: batch size per inferenza,
        - max_length: lunghezza massima sequenza (troncamento),
        - device: 'cuda' o 'cpu', se None sceglie automaticamente,
        - normalize: se True applica L2-normalization agli embeddings.

    Returns:
        (embeddings_data, info) dove:
            - embeddings_data['embeddings'] ha shape (n_items, hidden_dim),
            - info è un dict con metadati (model_name, device, max_length,
            normalize).

    Raises:
        ImportError: se 'torch' o 'transformers' non sono installati.

    Side Effects:
        stampa modello, device e shape finale a terminale.
    '''

    if (AutoTokenizer is None) or (AutoModel is None) or (torch is None):
        raise ImportError(f"transformers/torch non disponibili: installa "
                          f"'transformers' e 'torch'.")
    docs = _concat_text_cols(text_df, text_columns)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print('\n### TRANSFORMER EMBEDDINGS (HF)')
    print(f'- model: {model_name}')
    print(f'- device: {device}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            enc = tokenizer(batch_docs,
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                            return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            pooled = _mean_pooling(out.last_hidden_state, enc['attention_mask'])
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            all_vecs.append(pooled.cpu().numpy())
    X = np.vstack(all_vecs).astype(np.float32)
    embeddings_data = {'parent_asin': text_df[id_col].values,
                       'embeddings': X}
    info = {'model_name': model_name,
            'device': device,
            'max_length': max_length,
            'normalize': normalize}
    print(f'- X shape: {X.shape}')

    return embeddings_data, info


def save_embeddings_npz(path: str,
                        embeddings_data: Dict[str, np.ndarray]
                        ) -> None:
    
    print(f'\nSalvo embeddings su: {path}')
    np.savez_compressed(path, **embeddings_data)
    print('OK')


def load_embeddings_npz(path: str) -> Dict[str, np.ndarray]:

    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}



# 3) ANALISI EMBEDDINGS

def analyze_embeddings_detailed(embeddings_data: Dict[str, np.ndarray],
                                name: str,
                                topk_demo: int = 5,
                                n_demo_items: int = 3,
                                random_state: int = 42
                                ) -> None:
    '''
    Analizza embeddings in modo "utile" per un content-based RS.

    Controlla:
      1) Sanity checks: NaN/Inf, varianza nulla, duplicati esatti.
      2) Geometria: norme L2, cosine similarity media, anisotropia
         (PCA first component ratio).
      3) Struttura: PCA explained variance (prime componenti),
         'effective dimension' grezza.
      4) Retrieval demo: per alcuni item, stampa i nearest neighbors per cosine.

    Args:
        - embeddings_data: dict con chiavi:
            - 'parent_asin': (n_items,)
            - 'embeddings': (n_items, dim)
        - name: etichetta per output,
        - topk_demo: numero di nearest neighbors da mostrare nel demo,
        - n_demo_items: quanti item casuali usare per demo,
        - random_state: seed per selezione item demo.

    Returns:
        - None.
    
    Side Effects:
        - stampa le statistiche a terminale.
    '''

    item_ids = embeddings_data.get('parent_asin')
    X = embeddings_data.get('embeddings')

    print(f'\n\n==============================')
    print(f'EMBEDDINGS ANALYSIS: {name}')
    print(f'==============================')

    if (item_ids is None) or (X is None):
        print(f'[ERROR] embeddings_data deve contenere \'parent_asin\' '
              f'e \'embeddings\'.')
        return

    X = np.asarray(X)
    n, d = X.shape
    print(f'- n_items: {n:,} | dim: {d:,} | dtype: {X.dtype}')


    # 1) Sanity checks.

    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    print(f'\n[Sanity]')
    print(f'- NaN count: {n_nan:,} | Inf count: {n_inf:,}')


    # Varianza per dimensione (features "morte").

    var_dims = X.var(axis=0)
    zero_var = int((var_dims == 0).sum())
    print(f'- dims with zero variance: {zero_var:,}/{d:,}')

    rng = np.random.default_rng(random_state)
    sample_n = min(n, 10_000)
    idx_sample = rng.choice(n, size=sample_n, replace=False)
    Xs = X[idx_sample]
    rounded = np.round(Xs, 6)
    _any, unique_counts = np.unique(rounded, axis=0, return_counts=True)
    dup_rows = int((unique_counts > 1).sum())
    print(f'- exact/near-duplicate rows in sample({sample_n}): {dup_rows:,}')


    # 2) Geometria: norme e cosine stats.

    norms = np.linalg.norm(X, axis=1)
    print(f'\n[Geometry]')
    print(f'- L2 norm: mean={norms.mean():.4f} std={norms.std():.4f} '
          f'min={norms.min():.4f} max={norms.max():.4f}')
    pairs = 20_000
    if n >= 2:
        a = rng.integers(0, n, size=pairs)
        b = rng.integers(0, n, size=pairs)
        mask = a != b
        a, b = a[mask], b[mask]
        Xa, Xb = X[a], X[b]
        denom = (np.linalg.norm(Xa, axis=1) *
                 np.linalg.norm(Xb, axis=1) +
                 1e-9)
        cos = (Xa * Xb).sum(axis=1) / denom
        print(f'- cosine random-pairs: mean={cos.mean():.4f} '
              f'std={cos.std():.4f} p05={np.quantile(cos,0.05):.4f} '
              f'p95={np.quantile(cos,0.95):.4f}')
    try:
        from sklearn.decomposition import PCA
        pca_dims = min(50, d, n-1)
        X_center = X - X.mean(axis=0, keepdims=True)
        pca = PCA(n_components=pca_dims,
                  random_state=random_state)
        pca.fit(X_center)
        evr = pca.explained_variance_ratio_
        print(f'- PCA: EVR[PC1]={evr[0]:.4f} '
              f'| sum(EVR[:10])={evr[:10].sum():.4f} '
              f'| sum(EVR[:{pca_dims}])={evr.sum():.4f}')
        p = evr / (evr.sum() + 1e-12)
        eff_dim = float(1.0 / np.sum(p**2))
        print(f'- effective dimension (approx): {eff_dim:.2f}')
    except Exception as e:
        print(f'- PCA skipped: {e}')


    # 3) sparsity (utile soprattutto per TF-IDF).

    zero_ratio = float((X == 0).sum() / X.size)
    print(f'\n[Density]')
    print(f'- sparsity: {zero_ratio*100:.2f}% zeros')


    # 4) Retrieval demo: nearest neighbors.

    print(f'\n[Retrieval demo - cosine KNN]')
    try:
        from sklearn.neighbors import NearestNeighbors
        Xn = (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9))
        nn = NearestNeighbors(n_neighbors=min(topk_demo + 1, n),
                              metric='cosine')
        nn.fit(Xn)
        demo_idx = rng.choice(n,
                              size=min(n_demo_items, n),
                              replace=False)
        for q in demo_idx:
            dist, ind = nn.kneighbors(Xn[q:q+1],
                                      return_distance=True)
            dist = dist.flatten()
            ind = ind.flatten()
            ind = ind[1:]
            sim = 1.0 - dist[1:]
            print(f'\nQuery item: {item_ids[q]}')
            for rank, (j, s) in enumerate(zip(ind, sim), start=1):
                print(f'  {rank:02d}) {item_ids[j]}  sim={s:.4f}')
    except Exception as e:
        print(f'- retrieval demo skipped: {e}')
    
    return 


def analyze_embeddings(embeddings_data: Dict[str, np.ndarray],
                       name: str
                       ) -> None:
    '''
    Stampa statistiche descrittive degli embeddings (diagnostica rapida).

    Args:
        - embeddings_data: dict con almeno:
            - "parent_asin": array id item
            - "embeddings": matrice embeddings
        - name: nome del metodo (per stampa leggibile a terminale).

    Returns:
        - None

    Side Effects:
        - Stampa statistiche a terminale.
    '''

    print(f'\n\n### ANALISI EMBEDDINGS: {name}')
    ids = embeddings_data.get('parent_asin')
    X = embeddings_data.get('embeddings')
    if ids is None or X is None:
        print(f'[WARN] embeddings_data non contiene chiavi attese '
              f'(\'parent_asin\', \'embeddings\').')
        return

    print(f'- n_items: {len(ids):,}')
    print(f'- shape: {X.shape}')
    print(f'- mean: {float(X.mean()):.6f} | std: {float(X.std()):.6f}')
    print(f'- min:  {float(X.min()):.6f} | max: {float(X.max()):.6f}')
    zero_ratio = float((X == 0).sum() / X.size)
    print(f'- sparsity: {zero_ratio*100:.2f}% zeros')



def compute_kmeans_inertia_silhouette(embeddings_data: Dict[str, np.ndarray],
                                      k_list: List[int],
                                      embedding_name: str,
                                      random_state: int = 42,
                                      max_points: int = 20_000,
                                      sample_idx: Optional[np.ndarray] = None
                                      ) -> pd.DataFrame:
    '''
    Calcola inertia (elbow) e silhouette score per KMeans sugli embeddings.

    Perché serve:
        - Diagnostica della 'clusterability' dello spazio embedding.
        - Confronto tra diversi tipi di embeddings (TF-IDF vs CBoW 
          vs Transformer).

    Args:
        - embeddings_data: dict con chiave 'embeddings' (n_items, dim),
        - k_list: lista di k (= n_clusters) da provare,
        - embedding_name: nome per etichettare i risultati,
        - random_state: seed,
        - max_points: massimo numero di punti usati (campiona se troppo grande),
        - sample_idx: indici predefiniti da usare (consigliato per confronto
          equo tra embeddings).

    Returns:
        - DataFrame con colonne: ['embedding', 'k', 'inertia', 'silhouette',
          'n_points'].
    '''

    X = np.asarray(embeddings_data['embeddings'], dtype=np.float32)
    n = X.shape[0]

    if sample_idx is not None:
        Xs = X[sample_idx]
    else:
        if n > max_points:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n, size=max_points, replace=False)
            Xs = X[idx]
        else:
            Xs = X

    results = []
    for k in k_list:
        if k <= 1 or k >= Xs.shape[0]:
            results.append(
                {'embedding': embedding_name,
                 'k': k,
                 'inertia': np.nan,
                 'silhouette': np.nan,
                 'n_points': Xs.shape[0]}
            )
            continue
        km = KMeans(n_clusters=k,
                    n_init='auto',
                    random_state=random_state)
        labels = km.fit_predict(Xs)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(Xs, labels, metric='euclidean'))
        results.append(
            {'embedding': embedding_name,
             'k': k,
             'inertia': inertia,
             'silhouette': sil,
             'n_points': Xs.shape[0]}
        )

    return pd.DataFrame(results)


def plot_elbow_and_silhouette_single(df: pd.DataFrame,
                                     embedding_name: str
                                     ) -> None:
    '''
    Plotta elbow (inertia) e silhouette per un (solo) embedding.
    '''

    d = df[df['embedding'] == embedding_name].copy()

    plt.figure()
    plt.plot(d['k'], d['inertia'], marker='o')
    plt.title(f'Elbow (KMeans inertia) - {embedding_name}')
    plt.xlabel('k (n_clusters)')
    plt.ylabel('inertia')
    plt.grid(True)
    #plt.show()

    plt.figure()
    plt.plot(d['k'], d['silhouette'], marker='o')
    plt.title(f'Silhouette score (KMeans) - {embedding_name}')
    plt.xlabel('k (n_clusters)')
    plt.ylabel('silhouette')
    plt.grid(True)
    plt.show()



def plot_elbow_and_silhouette_comparison(df_all: pd.DataFrame) -> None:
    '''
    Plotta confronto tra embeddings: tre (o più) linee per inertia e silhouette.
    Matplotlib usa automaticamente colori diversi per ogni linea.
    '''

    embeddings = sorted(df_all['embedding'].unique().tolist())

    plt.figure()
    for emb in embeddings:
        d = df_all[df_all['embedding'] == emb]
        plt.plot(d['k'], d['inertia'], marker='o', label=emb)
    plt.title('Elbow comparison (KMeans inertia)')
    plt.xlabel('k (n_clusters)')
    plt.ylabel('inertia')
    plt.grid(True)
    plt.legend()
    #plt.show()

    plt.figure()
    for emb in embeddings:
        d = df_all[df_all['embedding'] == emb]
        plt.plot(d['k'], d["silhouette"], marker='o', label=emb)
    plt.title('Silhouette comparison (KMeans)')
    plt.xlabel('k (n_clusters)')
    plt.ylabel('silhouette')
    plt.grid(True)
    plt.legend()
    plt.show()


# 4) CONTENT-BASED ITEM-ITEM KNN

@dataclass
class ContentKNNArtifacts:
    item_ids: np.ndarray    # parent_asin array
    item_index: Dict[Any, int]  # asin -> idx
    neighbor_idx: np.ndarray    # (n_items, k)
    neighbor_sim: np.ndarray    # (n_items, k)


def build_item_knn_index(embeddings_data: Dict[str, np.ndarray],
                         n_neighbors: int,
                         metric: str = 'cosine'
                         ) -> ContentKNNArtifacts:
    '''
    Costruisce un indice KNN item-item sugli embeddings.

    Il content-based KNN richiede per ogni item i suoi item più simili.
    Precomputiamo questi vicini per:
        - accelerare predizione e valutazione
        - rendere ripetibile grid-search (cambiando solo k / metrica)

    Args:
        - embeddings_data: dict con 'parent_asin' e 'embeddings'.
        - n_neighbors: numero di vicini da mantenere per item (k).
        - metric: metrica per NearestNeighbors (default: "cosine").

    Returns:
        - product: un oggetto ContentKNNArtifacts contenente:
            - item_ids: array degli id item
            - item_index: dict asin -> col index
            - neighbor_idx: matrice (n_items, k) degli indici dei vicini
            - neighbor_sim: matrice (n_items, k) delle similarità (cosine-like)

    Side Effects:
        - Stampa dimensioni e parametri a terminale.

    Notes:
        - per cosine, sklearn ritorna distanza = 1 - cosine_sim; qui convertiamo
          in similarità.
        - converte distance -> similarity (per cosine: dist = 1 - cos_sim)
    '''

    item_ids = embeddings_data['parent_asin']
    X = embeddings_data['embeddings']
    if n_neighbors >= len(item_ids):
        n_neighbors = max(1, len(item_ids) - 1)
    print('\n### BUILD ITEM-KNN INDEX')
    print(f'- items: {len(item_ids):,}')
    print(f'- n_neighbors: {n_neighbors}')
    print(f'- metric: {metric}')
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1,
                          metric=metric,
                          algorithm="auto")
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)


    # remove self-neighbor (prima colonna)

    neighbor_idx = indices[:, 1:]
    neighbor_dist = distances[:, 1:]

    if metric == 'cosine':
        neighbor_sim = (1.0 - neighbor_dist).astype(np.float32)
    else:
        neighbor_sim = (1.0 / (1.0 + neighbor_dist)).astype(np.float32)

    item_index = {item_ids[i]: i for i in range(len(item_ids))}

    product = ContentKNNArtifacts(item_ids=item_ids,
                                  item_index=item_index,
                                  neighbor_idx=neighbor_idx,
                                  neighbor_sim=neighbor_sim)
    
    return product



def _build_sparse_ratings_matrix(ratings_df: pd.DataFrame,
                                 user_col: str = 'user_id',
                                 item_col: str = 'parent_asin',
                                 rating_col: str = 'rating',
                                 item_index: Optional[Dict[Any, int]] = None
                                 ) -> Tuple[csr_matrix, np.ndarray,
                                            Dict[Any, int]]:
    '''
    Costruisce una matrice sparse user-item delle valutazioni (CSR).

    Perché serve:
        La predizione content-based usa le valutazioni storiche dell'utente.
        Una matrice sparse CSR è:
            - molto più efficiente in memoria rispetto a una matrice densa
            - veloce da iterare per user history (indptr/indices/data)

    Args:
        - ratings_df: dataFrame con colonne user, item, rating.
        - user_col: nome colonna utente.
        - item_col: nome colonna item.
        - rating_col: nome colonna rating.
        - item_index: mapping item->col (se fornito, forza allineamento agli
          item degli embeddings).

    Returns:
        - (R, user_ids, user_index) dove:
            - R: csr_matrix shape (n_users, n_items)
            - user_ids: array degli id utenti (ordine righe)
            - user_index: dict user_id -> row index

    Notes:
        - Filtra automaticamente le righe con item non presenti in item_index
          (se fornito).
    '''

    df = ratings_df[[user_col, item_col, rating_col]].copy()
    df = df.dropna()
    user_ids = df[user_col].unique()
    user_index = {u: i for i, u in enumerate(user_ids)}
    if item_index is None:
        item_ids = df[item_col].unique()
        item_index = {it: i for i, it in enumerate(item_ids)}
    df = df[df[item_col].isin(item_index.keys())].copy()
    rows = df[user_col].map(user_index).to_numpy()
    cols = df[item_col].map(item_index).to_numpy()
    vals = df[rating_col].astype(np.float32).to_numpy()
    R = csr_matrix((vals, (rows, cols)),
                   shape=(len(user_ids),
                   len(item_index)),
                   dtype=np.float32)

    return R, user_ids, user_index


def predict_single(user_row: int,
                   item_col: int,
                   R: csr_matrix,
                   neighbors_idx_row: np.ndarray,
                   neighbors_sim_row: np.ndarray,
                   min_common: int = 1,
                   fallback: float = 3.5
                   ) -> float:
    '''
    Predice il rating di un singolo utente per un singolo item usando item-item
    KNN (content-based).

    Perché serve:
        - È il 'cuore' del modello: data la history utente e i vicini dell'item,
          calcola una predizione tramite media pesata dalle similarità.

    Formula:
        pred(u, i) = sum_j sim(i, j) * r(u, j) / sum_j |sim(i, j)|
        per j nei vicini dell'item che l'utente ha valutato.

    Args:
        - user_row: indice riga utente nella matrice CSR 'R'.
        - item_col: indice colonna item nella matrice CSR 'R'.
        - R: matrice sparse user-item delle valutazioni di TRAIN.
        - neighbors_idx_row: indici dei vicini dell'item (array di lunghezza k).
        - neighbors_sim_row: similarità associate (array di lunghezza k).
        - min_common: numero minimo di vicini 'in comune' (rated dall'utente)
          per accettare la predizione.
        - fallback: valore di fallback se non ci sono abbastanza informazioni
          (es. mean rating).

    Returns:
        - rating predetto (float).

    Notes:
        - 'item_col' è mantenuto per chiarezza API, anche se la predizione usa i
          vicini già precomputati.
    '''

    start, end = R.indptr[user_row], R.indptr[user_row + 1]
    rated_cols = R.indices[start:end]
    rated_vals = R.data[start:end]
    if rated_cols.size == 0:
        return fallback

    rated_map = dict(zip(rated_cols.tolist(), rated_vals.tolist()))
    num = 0.0
    den = 0.0
    common = 0
    for j, sim in zip(neighbors_idx_row, neighbors_sim_row):
        if j in rated_map:
            num += float(sim) * float(rated_map[j])
            den += abs(float(sim))
            common += 1
    if common < min_common or den == 0.0:
        return fallback
    
    return num / den


def evaluate_content_knn(ratings_df: pd.DataFrame,
                         artifacts: ContentKNNArtifacts,
                         test_size: float = 0.2,
                         random_state: int = 42,
                         min_common: int = 1,
                         fallback: float = 3.5
                         ) -> Dict[str, float]:
    '''
    Valuta il content-based item-item KNN con uno split train/test.

    Serve perchè abbiamo bisogno di un numero (RMSE/MAE) per:
            - fare grid search su k,
            - confrontare TF-IDF vs CBoW vs Transformer,
            - confrontare content-based vs collaborative (Surprise).

    Args:
        - ratings_df: dataFrame con user_id, parent_asin, rating,
        - artifacts: struttura contenente indice item e vicini (output di
          build_item_knn_index),
        - test_size: percentuale di test split,
        - random_state: seed per split riproducibile,
        - min_common: minimo numero di vicini valutati dall'utente per predire
          senza fallback,
        - fallback: valore base in caso di predizione non possibile (se train
          non disponibile verrà usato questo).

    Returns:
        - results_dict: un dict con:
            - 'rmse': Root Mean Squared Error sul test,
            - 'mae':  Mean Absolute Error sul test,
            - 'n_test_used': numero di esempi effettivamente valutati.

    Side Effects:
        - stampa a terminale quanti esempi sono stati usati e i punteggi.
    '''

    train_df, test_df = train_test_split(ratings_df,
                                         test_size=test_size,
                                         random_state=random_state)
    R_train, _uid, user_index = _build_sparse_ratings_matrix(
        train_df,
        item_index=artifacts.item_index
        )
    if len(train_df) > 0:
        fallback = float(train_df['rating'].mean())
    y_true = []
    y_pred = []
    missed_user = 0
    missed_item = 0
    for row in test_df.itertuples(index=False):
        u = getattr(row, 'user_id')
        it = getattr(row, 'parent_asin')
        r = float(getattr(row, 'rating'))

        if u not in user_index:
            missed_user += 1
            continue
        if it not in artifacts.item_index:
            missed_item += 1
            continue
        urow = user_index[u]
        icol = artifacts.item_index[it]
        pred = predict_single(
            user_row=urow,
            item_col=icol,
            R=R_train,
            neighbors_idx_row=artifacts.neighbor_idx[icol],
            neighbors_sim_row=artifacts.neighbor_sim[icol],
            min_common=min_common,
            fallback=fallback,
        )
        y_true.append(r)
        y_pred.append(pred)

    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    rmse = (float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) 
            else np.nan)
    mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else np.nan
    print('\n### EVAL CONTENT-KNN (train/test)')
    print(f'- test used: {len(y_true):,} | skipped users: {missed_user:,} '
          f'| skipped items: {missed_item:,}')
    print(f'- RMSE: {rmse:.4f} | MAE: {mae:.4f}')
    results_dict = {'rmse': rmse, 'mae': mae, 'n_test_used': float(len(y_true))}

    return results_dict


def grid_search_content_knn(ratings_df: pd.DataFrame,
                            embeddings_data: Dict[str, np.ndarray],
                            k_list: List[int],
                            metric: str = 'cosine',
                            test_size: float = 0.2,
                            random_state: int = 42,
                            min_common: int = 1
                            ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    '''
    Esegue una grid search semplice su k per il content-based item-item KNN.

    Args:
        - ratings_df: DataFrame ratings (user, item, rating),
        - embeddings_data: Dict embeddings ('parent_asin', 'embeddings'),
        - k_list: Lista di k (numero vicini) da provare,
        - metric: Metrica KNN (default: cosine),
        - test_size: Percentuale di test split per valutazione,
        - random_state: Seed per split riproducibile,
        - min_common: Minimo numero di vicini 'utili' (rated) per usare
          predizione pesata.

    Returns:
        - (results_df, best_params) dove:
            - results_df: DataFrame con colonne: k, rmse, mae, time_sec ordinato
              per rmse.
            - best_params: dict con i parametri migliori, es:
              {'n_neighbors': ..., 'metric': ..., 'rmse': ...}

    Side Effects:
        Stampa la tabella risultati e i best params a terminale.
    '''

    results = []
    best = {'rmse': np.inf, 'k': None}
    for k in k_list:
        start = time.time()
        artifacts = build_item_knn_index(embeddings_data,
                                         n_neighbors=k,
                                         metric=metric)
        scores = evaluate_content_knn(ratings_df=ratings_df,
                                      artifacts=artifacts,
                                      test_size=test_size,
                                      random_state=random_state,
                                      min_common=min_common)
        elapsed = time.time() - start

        row = {'k': k,
               'rmse': scores['rmse'],
               'mse': scores['mse'],
               'time_sec': elapsed}
        results.append(row)
        if scores['rmse'] < best['rmse']:
            best = {'rmse': scores['rmse'], 'k': k}

    df_results = pd.DataFrame(results).sort_values('rmse', ascending=True)
    print('\n### GRID SEARCH RESULTS')
    print(df_results)
    best_params = {'n_neighbors': int(best['k']),
                   'metric': metric,
                   'rmse': float(best['rmse'])}
    print(f'\nBest params: {best_params}')

    return df_results, best_params



def build_and_save_prediction_matrix_memmap(
    ratings_df: pd.DataFrame,
    artifacts: ContentKNNArtifacts,
    out_path_npy: str,
    chunk_items: int = 500,
    min_common: int = 1,
    fallback: Optional[float] = None,
) -> str:
    '''
    Costruisce e salva su disco la prediction matrix completa (users x items)
    usando memmap.

    Perché serve:
        - nel progetto intermedio vuoi salvare le recommendation/prediction 
          matrices per evitare di ricomputarle ogni volta.
        - la matrice completa può essere enorme, usare memmap + chunk:
            - evita di tenere tutto in RAM,
            - permette di scrivere incrementale su file .npy.

    Args:
        - ratings_df: dataFrame ratings (user_id, parent_asin, rating) usato
          come train,
        - artifacts: output di build_item_knn_index
          (vicini + mapping item->index),
        - out_path_npy: path di output. Se non termina con '.npy' viene aggiunto
          automaticamente,
        - chunk_items: numero di item per chunk di elaborazione (trade-off
          tempo/memoria),
        - min_common: minimo numero di vicini valutati dall'utente per predire
          senza fallback,
        - fallback: valore di fallback. Se None, usa media rating del dataset
          (train mean).

    Returns:
        - path del file '.npy' scritto su disco.

    Side Effects:
        - scrive su disco (memmap) e stampa i progressi a terminale.
    '''

    if fallback is None:
        fallback = (float(ratings_df['rating'].mean()) if len(ratings_df)
                    else 3.5)

    R, _uid, _iid = _build_sparse_ratings_matrix(
        ratings_df,
        item_index=artifacts.item_index
        )
    n_users, n_items = R.shape

    print('\n### BUILD FULL PREDICTION MATRIX (memmap)')
    print(f'- users: {n_users:,} | items: {n_items:,}')
    print(f'- chunk_items: {chunk_items}')
    print(f'- fallback: {fallback:.4f}')


    # memmap file

    out_file = (out_path_npy if out_path_npy.endswith('.npy')
                else out_path_npy + '.npy')
    pred_mm = np.memmap(out_file,
                        dtype=np.float32,
                        mode="w+",
                        shape=(n_users, n_items))

    user_rated_cols = []
    user_rated_vals = []
    for u in range(n_users):
        start, end = R.indptr[u], R.indptr[u + 1]
        user_rated_cols.append(R.indices[start:end])
        user_rated_vals.append(R.data[start:end])

    for i0 in range(0, n_items, chunk_items):
        i1 = min(n_items, i0 + chunk_items)
        print(f'  - items chunk: [{i0}, {i1})')
        for i in range(i0, i1):
            neigh_idx = artifacts.neighbor_idx[i]
            neigh_sim = artifacts.neighbor_sim[i]
            col_preds = np.full((n_users,), fallback, dtype=np.float32)
            for u in range(n_users):
                cols = user_rated_cols[u]
                vals = user_rated_vals[u]
                if cols.size == 0:
                    continue
                rated_map = dict(zip(cols.tolist(), vals.tolist()))
                num = 0.0
                den = 0.0
                common = 0
                for j, sim in zip(neigh_idx, neigh_sim):
                    if j in rated_map:
                        num += float(sim) * float(rated_map[j])
                        den += abs(float(sim))
                        common += 1
                if common >= min_common and den > 0.0:
                    col_preds[u] = num / den
            pred_mm[:, i] = col_preds
        pred_mm.flush()

    print(f'\nSaved prediction matrix: {out_file}')

    return out_file