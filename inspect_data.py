import pandas as pd
import numpy as np

df = pd.read_excel('dataset/Online Retail.xlsx')

print('SHAPE:', df.shape)
print('\nCOLUMNS:', df.columns.tolist())
print('\nDTYPES:')
print(df.dtypes.to_string())
print('\nHEAD:')
print(df.head().to_string())
print('\nMISSING:')
print(df.isnull().sum().to_string())
print('\nDUPLICATES:', df.duplicated().sum())
print('\nDESCRIBE:')
print(df.describe().to_string())
print('\nUNIQUE PER COL:')
for col in df.columns:
    print(f'  {col}: {df[col].nunique()} unique')
print('\nNEGATIVES/ZEROS:')
for col in df.select_dtypes(include=[np.number]).columns:
    print(f'  {col}: neg={(df[col]<0).sum()}, zero={(df[col]==0).sum()}')
print('\nOUTLIERS (IQR method):')
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    print(f'  {col}: {out} outliers')
