import pandas as pd

xl = pd.ExcelFile('dataset/Online Retail.xlsx')
print('Sheets:', xl.sheet_names)

df = pd.read_excel('dataset/Online Retail.xlsx')

print('Cancelled invoices (C prefix):', df['InvoiceNo'].astype(str).str.startswith('C').sum())
print('Date range:', df['InvoiceDate'].min(), 'to', df['InvoiceDate'].max())
print('Rows without CustomerID:', df['CustomerID'].isna().sum())

print('\nSample negative Quantity rows:')
print(df[df['Quantity']<0][['InvoiceNo','Quantity','UnitPrice','CustomerID']].head(5).to_string())

print('\nSample zero/negative UnitPrice rows:')
print(df[df['UnitPrice']<=0][['InvoiceNo','UnitPrice','Description']].head(5).to_string())

print('\nCountry distribution (top 10):')
print(df['Country'].value_counts().head(10).to_string())
