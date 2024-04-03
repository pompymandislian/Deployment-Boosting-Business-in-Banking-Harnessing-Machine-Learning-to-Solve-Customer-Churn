import pytest
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from SeparateOutputInput import SeparateOutputInput

# Import data
data_url = "https://raw.githubusercontent.com/pompymandislian/learn_CI_CD/main/data.csv"
response = requests.get(data_url)

# Simpan response content ke dalam file sementara
with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
    tmp_file.write(response.content)

# Baca file sementara menggunakan pandas
data = pd.read_csv(tmp_file.name, sep=';')

# Hapus file sementara setelah selesai membacanya
import os
os.unlink(tmp_file.name)

def test_start_end_columns():
    assert data.columns.any() == 'credit_score' # check start columns
    assert data.columns.all() == 'Age' # check final columns

def test_dtypes():
    assert data['credit_score'].dtype == 'int64'
    assert data['country'].dtype == 'object'
    assert data['gender'].dtype == 'object'
    assert data['tenure'].dtype == 'int64'
    assert data['balance'].dtype == 'float64'
    assert data['credit_card'].dtype == 'int64'
    assert data['active_member'].dtype == 'int64'
    assert data['estimated_salary'].dtype == 'float64'
    assert data['churn'].dtype == 'int64'
    assert data['Age'].dtype == 'object'

# SeparateOutputInput 
X, y = SeparateOutputInput(data = data,
                          output_column_name = "churn")

def test_spliting():
    # Fungsi train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=123, stratify=y)
    # Assertion untuk memastikan jumlah kolom setelah pemisahan
    assert X_train.shape[1] == 9
    assert X_test.shape[1] == 9


