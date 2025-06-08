# hotel_preprocessing.py

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

def impute_median(series):
    #print(series.median())
    return series.fillna(series.median())

def clean_data(df):
    df = df.copy()
    df['children'] = df['children'].transform(impute_median)  # Assumes you have this function defined
    df['agent'] = df['agent'].astype('object').fillna('Not Specified')
    df['country'] = df['country'].fillna(str(df['country'].mode()[0]))
    df.drop_duplicates(inplace=True)
    return df

def extract_features(df):
    df = df.copy()

    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_date_month'] + '-' +
        df['arrival_date_day_of_month'].astype(str)
    )

    latest_date = df['arrival_date'].max()
    three_months_ago = latest_date - timedelta(days=90)
    df['arrival_date_in_3_month'] = (df['arrival_date'] >= three_months_ago).astype(int)

    #print(f"Percentage of data within the last 3 months: {(df['arrival_date_in_3_month'].mean() * 100):.2f}%")

    #df['is_canceled'] = df['is_canceled'].astype(int)
    df['lead_time_log'] = np.log1p(df['lead_time'])
    df['booking_changes_log'] = np.log1p(df['booking_changes'])

    season_map = {
        'December': 'Winter', 'January': 'Winter', 'February': 'Winter',
        'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
        'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
        'September': 'Fall', 'October': 'Fall', 'November': 'Fall'
    }
    df['arrival_date_season'] = df['arrival_date_month'].map(season_map)

    meal = {'BB': 1, 'HB': 1, 'FB': 1, 'SC': 0, 'Undefined': 0}
    df['meal_bin'] = df['meal'].apply(lambda x: meal.get(x, 0))

    df['total_guests'] = df['adults'] + df['children'].astype(int) + df['babies']
    df['room_type_match'] = (df['reserved_room_type'] == df['assigned_room_type']).astype(int)

    # Top N binning (calculate based on current df, not external hotel_data)
    top_N_countries = df['country'].value_counts().nlargest(10).index
    df['country_binned'] = df['country'].apply(lambda x: x if x in top_N_countries else 'Other')
    #print(f"Top 10 Frequent Countries: {list(top_N_countries)}")

    top_N_agents = df['agent'].value_counts().nlargest(5).index
    df['agent_binned'] = df['agent'].apply(lambda x: x if x in top_N_agents else 'Other')
    #print(f"Top 5 Frequent Agents: {list(top_N_agents)}")

    return df

def drop_columns(df):
    df = df.copy()
    dropped_cols = [
        'company', 'agent', 'arrival_date_week_number',
        'reservation_status_date', 'reservation_status',
        'arrival_date',
        'meal', 'country'
    ]
    df.drop(dropped_cols, axis=1, inplace=True, errors='ignore')
    return df

class XYPreprocessor:
    def __init__(self, drop_first=False):
        self.drop_first = drop_first
        self.scaler = MinMaxScaler()
        self.numeric_cols = []
        self.cat_cols = []
        self.fitted_columns = None  # store full column set after encoding

    def fit(self, X, y):
        X, y = self._apply_cleaning(X, y)

        self.numeric_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object']).columns.tolist()

        self.scaler.fit(X[self.numeric_cols])

        # Apply get_dummies to capture all possible one-hot columns
        X_scaled = X.copy()
        X_scaled[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        X_encoded = pd.get_dummies(X_scaled, columns=self.cat_cols, drop_first=self.drop_first, dtype='int')

        self.fitted_columns = X_encoded.columns  # save full column set
        return self

    def transform(self, X, y=None):
        if y is not None:
            X, y = self._apply_cleaning(X, y)
            y = y.loc[X.index]
        else:
            X = self._apply_cleaning(X)

        X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        X = pd.get_dummies(X, columns=self.cat_cols, drop_first=self.drop_first, dtype='int')

        # Reindex to match training columns
        X = X.reindex(columns=self.fitted_columns, fill_value=0)

        if y is not None:
            return X, y
        else:
            return X

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)

    def _apply_cleaning(self, X, y=None):
        X = X.copy()
        X = clean_data(X)
        X = extract_features(X)
        X = drop_columns(X)
        if y is not None:
            y = y.loc[X.index]
            return X, y
        return X


