import pandas as pd


import quandl

df = quandl.get('WIKI/GOOGL')

print(df)

print("here in the middle ")
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

#high minus low
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

#percent change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

newdf= df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

print(df)

print(newdf)