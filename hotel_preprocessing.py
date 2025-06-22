import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class XYPreprocessor:
    def __init__(self, top_n_categories=10, top_n_dict=None):
        self.top_n_categories = top_n_categories
        self.top_n_dict = top_n_dict or {}
        self.top_categories_map = {}
        self.numeric_features = None
        self.categorical_features = None
        self.column_transformer = None
        self.feature_names_ = None

    def clean_data(self, df):
        df = df.copy()
        df['children'] = df['children'].fillna(df['children'].median())
        df['country'] = df['country'].fillna('Unknown')
        df['agent'] = df['agent'].astype('object').fillna('Not Specified')
        df['company'] = df['company'].fillna(0)
        df = df.drop_duplicates()
        df = df[~((df['adults'] == 0) & (df['children'] == 0) & (df['babies'] == 0))]
        df = df[~((df['stays_in_weekend_nights'] == 0) & (df['stays_in_week_nights'] == 0))]
        return df

    def extract_features(self, df):
        df = df.copy()
        df['arrival_date'] = pd.to_datetime(
            df['arrival_date_year'].astype(str) + '-' +
            df['arrival_date_month'].astype(str) + '-' +
            df['arrival_date_day_of_month'].astype(str), errors='coerce')
        df['total_guests'] = df['adults'] + df['children'] + df['babies']
        df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        df['arrival_date_month'] = pd.to_datetime(df['arrival_date_month'], format='%B').dt.month
        df['lead_time_log'] = np.log1p(df['lead_time'].fillna(0))
        df['booking_changes_log'] = np.log1p(df['booking_changes'].fillna(0))
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                      3: 'Spring', 4: 'Spring', 5: 'Spring',
                      6: 'Summer', 7: 'Summer', 8: 'Summer',
                      9: 'Fall', 10: 'Fall', 11: 'Fall'}
        df['arrival_date_season'] = df['arrival_date_month'].map(season_map)
        meal_map = {'BB': 1, 'HB': 1, 'FB': 1, 'SC': 0, 'Undefined': 0}
        df['meal_bin'] = df['meal'].apply(lambda x: meal_map.get(x, 0))
        df['room_type_match'] = (df['reserved_room_type'] == df['assigned_room_type']).astype(int)
        df['booking_weekday'] = pd.to_datetime(df['reservation_status_date'], errors='coerce').dt.dayofweek.map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'})
        df['arrival_weekend'] = df['arrival_date_day_of_month'].isin([5,6,12,13,19,20,26,27]).astype(int)
        # Fix mixed types for OneHotEncoder
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].astype(str)
        return df

    def drop_columns(self, df):
        cols_to_drop = [
            'company', 'arrival_date_week_number',
            'reservation_status_date', 'reservation_status',
            'arrival_date', 'meal',
            'assigned_room_type', 'reserved_room_type'
        ]
        return df.drop(columns=cols_to_drop)

    def clean_and_feature_engineer(self, df):
        df_clean = self.clean_data(df.copy())
        df_feat = self.extract_features(df_clean)
        return df_feat

    def fit(self, df, y):
        df_feat = self.clean_and_feature_engineer(df.copy())
        df_feat = df_feat.drop(columns=['is_canceled'], errors='ignore')
        y_aligned = y.loc[df_feat.index]
        df_features = self.drop_columns(df_feat)
        self.numeric_features = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df_features.select_dtypes(include=['object']).columns.tolist()
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        self.column_transformer = ColumnTransformer(transformers=[
            ("num", numeric_transformer, self.numeric_features),
            ("cat", categorical_transformer, self.categorical_features)
        ])
        self.column_transformer.fit(df_features, y_aligned)
        self.feature_names_ = self.column_transformer.get_feature_names_out()
        return self, df_features, y_aligned

    def transform(self, df):
        df_feat = self.clean_and_feature_engineer(df.copy())
        df_features = self.drop_columns(df_feat)
        transformed_array = self.column_transformer.transform(df_features)
        return pd.DataFrame(transformed_array, columns=self.feature_names_, index=df_features.index)


    def fit_transform(self, df, y):
        self, df_features, y_aligned = self.fit(df, y)
        transformed_array = self.column_transformer.transform(df_features)
        return pd.DataFrame(transformed_array, columns=self.feature_names_, index=df_features.index), y_aligned
