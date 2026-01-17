'''
Conversione dei dataset Amazon Reviews 2023 da formato .jsonl a .parquet.

Per motivi di performance e memoria, la conversione avviene a chunk:
  1) lettura progressiva da jsonl con pandas (chunksize)
  2) scrittura di ogni chunk in formato parquet
  3) (opzionale) ricaricamento e concatenazione dei chunk in un parquet finale

Dataset gestiti (categoria Books):
  - Books.jsonl -> Books.parquet
  - meta_Books.jsonl -> meta_Books.parquet (con pulizia prezzo)

Output:
  - file .parquet su disco (chunks + final)

Nota:
  Questo script è pensato per essere eseguito 'una tantum' durante il setup dei
  dati. Nel resto del progetto si lavora direttamente su parquet.
'''

from imports import *

input_file = "C:/Users/gcaso/Documents/progetti_uni/RS-MIGA/data/Books.jsonl"
reviews_output_dir = "C:/Users/gcaso/Documents/progetti_uni/RS-MIGA/data/parquet_chunks"
final_output = "Books.parquet"
DATA_PATH = "C:/Users/gcaso/Documents/progetti_uni/RS-MIGA/data/"
os.makedirs(reviews_output_dir, exist_ok=True)
chunksize = 500_000 
chunk_num = 0


# Salva il dataset in chunks .parquet

with pd.read_json(input_file, lines=True, chunksize=chunksize) as reader:
    for chunk in reader:
        chunk_file = os.path.join(reviews_output_dir,
                                  f"chunk_{chunk_num:03d}.parquet")
        chunk.to_parquet(chunk_file, engine="pyarrow", index=False)
        print(f"Salvato {chunk_file}")
        chunk_num += 1
print("Conversione a chunk completata.")


# Carica e unisce tutti i chunks .parquet

all_files = sorted(glob.glob(os.path.join(reviews_output_dir, "*.parquet")))
df_list = [pd.read_parquet(f) for f in all_files]
df_final = pd.concat(df_list, ignore_index=True)
df_final.to_parquet(final_output, engine="pyarrow", index=False)
print(f"Parquet finale salvato in {final_output}")

input_file = "C:/Users/gcaso/Documents/progetti_uni/RS-MIGA/data/meta_Books.jsonl"
meta_output_dir = "C:/Users/gcaso/Documents/progetti_uni/RS-MIGA/data/parquet_chunks"
final_output = "meta_Books.parquet"
os.makedirs(meta_output_dir, exist_ok=True)
chunksize = 250_000 
chunk_num = 0

def clean_price(val):
    '''
    Normalizza il valore 'price' (stringhe sporche) a float o None.

    Args:
        val: Valore originale della colonna 'price' (può essere NaN, stringa o
        numero).

    Returns:
        float: Prezzo ripulito.
        None: Se il valore è nullo o non parseable.
    '''
    if pd.isna(val):
        return None
    val = re.sub(r"[^0-9.]", "", str(val))
    return float(val) if val else None


# Salva il dataset in chunks .parquet

with pd.read_json(input_file, lines=True, chunksize=chunksize) as reader:
    for chunk in reader:
        chunk_file = os.path.join(meta_output_dir,
                                  f"chunk_{chunk_num:04d}.parquet")
        if "price" in chunk.columns:
            chunk["price"] = chunk["price"].apply(clean_price)
        chunk.to_parquet(chunk_file, engine="pyarrow", index=False)
        print(f"Salvato {chunk_file}")
        chunk_num += 1
print("Conversione a chunk completata.")


# Carica e unisce tutti i chunks .parquet

all_files = sorted(glob.glob(os.path.join(meta_output_dir, "*.parquet")))
df_list = [pd.read_parquet(f) for f in all_files]
df_final = pd.concat(df_list, ignore_index=True)
df_final.to_parquet(final_output, engine="pyarrow", index=False)
print(f"Parquet finale salvato in {final_output}")
all_files = sorted(glob.glob(os.path.join(meta_output_dir, "*.parquet")))


# Scrive il primo chunk come base

df_first = pd.read_parquet(all_files[0])
df_first.to_parquet(f'{DATA_PATH}meta_Books.parquet',
                    engine='pyarrow',
                    index=False)
print(f"Inizializzato file finale con {all_files[0]}")


# Itera sugli altri chunk

print(f"Aggiungo {all_files[4]}...")
df_final = pd.read_parquet(f'{DATA_PATH}meta_Books.parquet')
df_chunk = pd.read_parquet(all_files[4])
df_final = pd.concat([df_final, df_chunk], ignore_index=True)
df_final.to_parquet(f'{DATA_PATH}meta_Books.parquet',
                    engine='pyarrow',
                    index=False)
print(f"Aggiunto {all_files[4]} al file finale")
print(f"Parquet finale salvato in {final_output}")