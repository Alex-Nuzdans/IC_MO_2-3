import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class code:

    def load_data(name):
        df = pd.read_csv(name)
        return df

    def transform(df,target_column):
        categorical_features = []
        numeric_features = []

        for col,type in zip(df.columns, df.dtypes):
            if target_column is not None and col == target_column:
                continue
            if type=='float64'or type=='int64':
                numeric_features.append(col)
            else:
                categorical_features.append(col)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),  # для числовых
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features) # для категориальных
            ]
        )
        return preprocessor
