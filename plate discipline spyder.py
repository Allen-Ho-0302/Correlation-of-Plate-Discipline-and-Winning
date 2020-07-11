import pandas as pd
import pyodbc

sql_conn = pyodbc.connect('''DRIVER={ODBC Driver 13 for SQL Server};
                            SERVER=ALLENHO\MSSQLSERVER002;
                            DATABASE=Plate discipline and winning correlation;
                            Trusted_Connection=yes''') 
query = '''
SELECT p.name, ([Z-Swing%]-[O-Swing%])/[Swing%] as plated, [BB%], [K%], AVG, OBP, ISO, wOBA, [wRC+], (d.WAR/d.PA) as per_war
FROM ['plate discipline 2010-2019 1000$'] p
JOIN ['dashboard stats 2010-2019 1000P$'] d
on p.name = d.name
order by WAR desc;
'''
df = pd.read_sql(query, sql_conn)

df.columns = df.columns.astype(str)

df_new = df.loc[:, 'plated':'per_war']

df_corr = df_new.corr()

df_corr_new = df_corr.loc['BB%':'per_war',:]


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

print(df_corr_new.index)


fig, ax = plt.subplots()

ax.bar(df_corr_new.index, df_corr_new['plated'])

ax.set_xticklabels(df_corr_new.index, rotation=90)

ax.set_ylabel('correlation coefficent with plate discipline')

plt.show()



#---------------------------------------------------------------------------
#import statsmodels.formula.api as smf
#df['per_war2'] = df['per_war']**2

#df['wRC'] = df['wRC+']
#df['wRC2'] = df['wRC+']**2
#df['wOBA2'] = df['wOBA']**2
#df['ISO2'] = df['ISO']**2
#df['OBP2'] = df['OBP']**2
#df['AVG2'] = df['AVG']**2
#df['K'] = df['K%']
#df['K2'] = df['K%']**2
#df['BB'] = df['BB%']
#df['BB2'] = df['BB%']**2

#results = smf.ols('plated~per_war + per_war2 + wRC + wRC2 + wOBA + wOBA2 + ISO + ISO2 + OBP + OBP2 + AVG + AVG2 + K + K2 + BB + BB2', data=df).fit()

#print(results.params)

#df = pd.DataFrame()
#df['educ'] = np.linspace(0,20)
#df['age'] = 30
#df['educ2'] = df['educ']**2
#df['age2'] = df['age']**2

#pred = results.predict(df)
#print(pred.head())



#----------------------------------------------------------------------
#plt.plot(df_new['plated'], df_new['per_war'], 'o', alpha=0.2)

#res = linregress(df_new['plated'], df_new['per_war'])
#fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
#fy = res.intercept + res.slope * fx
#plt.plot(fx, fy)

#plt.xlabel('plate discipline')
#plt.ylabel('per_game_war')
#plt.show()
#----------------------------------------------------------------------
#plt.plot(df_new['plated'], df_new['wRC+'], 'o', alpha=0.2)

#res = linregress(df_new['plated'], df_new['wRC+'])
#fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
#fy = res.intercept + res.slope * fx
#plt.plot(fx, fy)

#plt.xlabel('plate discipline')
#plt.ylabel('wRC+')
#plt.show()
#-----------------------------------------------------------------------
#plt.plot(df_new['plated'], df_new['wOBA'], 'o', alpha=0.2)

#res = linregress(df_new['plated'], df_new['wOBA'])
#fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
#fy = res.intercept + res.slope * fx
#plt.plot(fx, fy)

#plt.xlabel('plate discipline')
#plt.ylabel('wOBA')
#plt.show()
#-----------------------------------------------------------------------
#plt.plot(df_new['plated'], df_new['ISO'], 'o', alpha=0.2)

#res = linregress(df_new['plated'], df_new['ISO'])
#fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
#fy = res.intercept + res.slope * fx
#plt.plot(fx, fy)

#plt.xlabel('plate discipline')
#plt.ylabel('ISO')
#plt.show()
#----------------------------------------------------------------------
#plt.plot(df_new['plated'], df_new['OBP'], 'o', alpha=0.2)

#res = linregress(df_new['plated'], df_new['OBP'])
#fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
#fy = res.intercept + res.slope * fx
#plt.plot(fx, fy)

#plt.xlabel('plate discipline')
#plt.ylabel('OBP')
#plt.show()
#----------------------------------------------------------------------
#plt.plot(df_new['plated'], df_new['AVG'], 'o', alpha=0.2)

#res = linregress(df_new['plated'], df_new['AVG'])
#fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
#fy = res.intercept + res.slope * fx
#plt.plot(fx, fy)

#plt.xlabel('plate discipline')
#plt.ylabel('AVG')
#plt.show()
#-----------------------------------------------------------------------
#plt.plot(df_new['plated'], df_new['K%'], 'o', alpha=0.2)

#res = linregress(df_new['plated'], df_new['K%'])
#fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
#fy = res.intercept + res.slope * fx
#plt.plot(fx, fy)

#plt.xlabel('plate discipline')
#plt.ylabel('K%')
#plt.show()
#-----------------------------------------------------------------------
#plt.plot(df_new['plated'], df_new['BB%'], 'o', alpha=0.2)

#res = linregress(df_new['plated'], df_new['BB%'])
#fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
#fy = res.intercept + res.slope * fx
#plt.plot(fx, fy)

#plt.xlabel('plate discipline')
#plt.ylabel('BB%')
#plt.show()
