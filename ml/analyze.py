import pandas as pd
import json
df = pd.read_csv('credit_risk_dataset.csv')
info = {
    'columns': df.columns.tolist(),
    'dtypes': {k: str(v) for k, v in df.dtypes.items()},
    'head': df.head(3).to_dict(orient='records'),
    'nulls': df.isnull().sum().to_dict()
}
with open('dataset_info.json', 'w') as f:
    json.dump(info, f, indent=2)
