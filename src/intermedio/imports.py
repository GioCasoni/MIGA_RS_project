from __future__ import annotations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
import dask.dataframe as dd
import gensim as gs
import pyarrow as pa
from pyarrow import parquet as pq
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split
from surprise.model_selection import GridSearchCV
from surprise import KNNBasic, SVD
from collections import Counter
from itertools import islice
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import warnings
import time
import re
import glob
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Tuple, Any, List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
