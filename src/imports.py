"""imports.py

Raccolta centralizzata di import usati nel progetto.

Questo modulo esiste per:
  - mantenere coerente lo stack di librerie tra i vari script
  - ridurre boilerplate e rendere più rapidi i notebook/script
  - semplificare l'ambiente di esecuzione (un solo punto da aggiornare)

Uso:
  In quasi tutti i file del progetto si usa:
      from imports import *

Note:
  - L'uso di wildcard import è intenzionale per mia semplicità, mi rendo conto
  che in progetto 'seri' sia preferibile import esplicito per evitare namespace
    pollution e migliorare la leggibilità.
"""

from __future__ import annotations
import warnings
import time
import re
import glob
import os
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import dask.dataframe as dd

import sklearn as skl
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


import gensim as gs
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

import pyarrow as pa
from pyarrow import parquet as pq

from surprise import Dataset, Reader, accuracy, KNNBasic, SVD
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV

from collections import Counter
from itertools import islice
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Tuple, Any, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")
