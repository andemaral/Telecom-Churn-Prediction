from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from itertools import product as iproduct
from functools import reduce
from operator import mul
from collections import Counter
import numpy as np


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, *_):
        return self
    
    def transform(self, X, *_):
        return X[self.columns]
    

class GetDummiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, drop_first=True):
        self.drop_first = drop_first
        if isinstance(columns, str):
            self.columns = [columns]
        else:
            self.columns = columns

    def transform(self, X, *_):
        if hasattr(self, 'dummies_cols_'):
            dummy_df = pd.DataFrame()
            for col in self.columns:
                dummies = self.dummies_cols_[
                    self.dummies_cols_.str.contains(col + '_')]
                dummies = dummies.str.replace(col + '_', '')
                for dummy in dummies:
                    series_d = pd.Series(X[col]).apply(
                        lambda x: 1 if x == dummy else 0)
                    dummy_df[col + '_' + dummy] = series_d
            return dummy_df
        else:
            if isinstance(X, pd.DataFrame):
                return pd.get_dummies(X, columns=self.columns,
                    drop_first=self.drop_first)
            else:
                raise TypeError(
                    "Este Transformador solo funciona en DF de Pandas"
                )

    def fit(self, X, *_):
        self.dummies_cols_ = pd.Series(pd.get_dummies(X[self.columns],
            columns=self.columns, drop_first=self.drop_first,
            prefix=[col for col in self.columns]).columns)
        return self


class OutlierTrim(BaseEstimator, TransformerMixin):
    def __init__(self, quant=0, cols=None):
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.quant = quant
        self.dict_ = None

    def fit(self, df, *_):
        self.dict_ = df.quantile([self.quant, 1 - self.quant]).to_dict()
        return self

    def transform(self, df, *_):
        if self.cols is None:
            self.cols = df.columns
        df_t = df[self.cols].copy()
        for col in self.cols:
            df_t.loc[:, col] = df_t.loc[:, col].clip(
                self.dict_[col][self.quant], self.dict_[col][1 - self.quant])

        return df_t


class PolyDictVectorizer(DictVectorizer):
    def __init__(self, degree=2, sparse=True, num_types=[float, np.float64]):
        self.degree = degree
        self.num_types = num_types
        super().__init__(sparse=sparse)

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_dict('records')
        X = [encode(x, self.degree, self.num_types) for x in X]
        return super().fit(X, y)

    def transform_to_pandas(self, X, conserve_index=True):
        if conserve_index:
            self.index = X.index
        if isinstance(X, pd.DataFrame):
            X = X.to_dict('records')
        X = [encode(x, self.degree, self.num_types) for x in X]
        return pd.DataFrame(X, index=self.index)

    def _transform(self, X, fitting):
        if isinstance(X, pd.DataFrame):
            X = X.to_dict('records')
        X = [encode(x, self.degree, self.num_types) for x in X]
        return super()._transform(X, fitting)


class PolyFeatureHasher(FeatureHasher):

    def __init__(self, degree=2, n_features=2**20, num_types=[float,
                                                              np.float64]):
        self.degree = degree
        self.num_types = num_types
        super().__init__(n_features=n_features)

    def transform(self, X):
        X = [encode(x, self.degree, self.num_types) for x in X]
        return super().transform(X)


def product(iterable, start=1):
    return reduce(mul, iterable, start)


def encode(dic, degree, num_types):
    dic = {k if type(v) in num_types else f'{k}={v}':
           float(v) if type(v) in num_types else 1
           for k, v in dic.items()}
    dic_keys = list(dic.keys())
    for deg in range(2, degree + 1):
        for term_keys in iproduct(dic_keys, repeat=deg):
            term_names, term_facts = [], []
            for k, n in Counter(term_keys).items():
                v = dic[k]
                if type(v) is int and n > 1:
                    break
                term_names.append(k if n == 1 else f'{k}^{n}')
                term_facts.append(v**n)
            else:  # No dummy feature was included more than once
                dic['*'.join(sorted(term_names))] = product(term_facts)
    return dic
