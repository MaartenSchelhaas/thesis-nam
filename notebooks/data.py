import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

data_path = r"datasets\raw\compas-scores-two-years.csv"

df = pd.read_csv(data_path)
#Length of stay removed, think thats a calculated feature based on two dates.
df = df[['age', 'race', 'sex', 'priors_count', 'c_charge_degree', 'two_year_recid']]

X = df.drop(columns=['two_year_recid'])
y = df['two_year_recid'].to_numpy()

categorical_cols = ['race', 'sex', 'c_charge_degree']
numerical_cols = ['age', 'priors_count']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(sparse_output=False), categorical_cols),
    ('num', MinMaxScaler((-1, 1)), numerical_cols),
])

X = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train, y_train)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(X_train.shape)