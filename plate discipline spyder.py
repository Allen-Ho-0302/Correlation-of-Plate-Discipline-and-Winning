import pandas as pd
import pyodbc

#connect with sql server, which already contains two tables from FanGraphs
#1.https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=1000&type=8&season=2019&month=0&season1=2010&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate=&enddate=
#2.https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=1000&type=5&season=2019&month=0&season1=2010&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate=2010-01-01&enddate=2019-12-31
sql_conn = pyodbc.connect('''DRIVER={ODBC Driver 13 for SQL Server};
                            SERVER=ALLENHO\MSSQLSERVER002;
                            DATABASE=Plate discipline and winning correlation;
                            Trusted_Connection=yes''') 

#sql query to grab data, including players from 2010-2019 with over 1000 PA.
#Define plate discipline as [Z-Swing%]-[O-Swing%])/[Swing%]
#Also grab data from that corresponding player's BB%, K%, AVG, OBP, ISO, wOBA, wRC+, WAR/PA for that period
query = '''
SELECT p.name, ([Z-Swing%]-[O-Swing%])/[Swing%] as plated, [BB%], [K%], AVG, OBP, ISO, wOBA, [wRC+], (d.WAR/d.PA) as per_war
FROM ['plate discipline 2010-2019 1000$'] p
JOIN ['dashboard stats 2010-2019 1000P$'] d
on p.name = d.name
order by WAR desc;
'''

#convert the data into dataframe
df = pd.read_sql(query, sql_conn)

#convert columns' type into string
df.columns = df.columns.astype(str)

#slice the data into only columns from plated(respresent plate discipline) to per_war(represent per game war)
df_new = df.loc[:, 'plated':'per_war']

#get the correlation table from df_new
df_corr = df_new.corr()

#slice the correlation table(df_corr) row from BB% to per game war
df_corr_new = df_corr.loc['BB%':'per_war',:]


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

fig, ax = plt.subplots()

#plot the correlation table as bar plot
ax.bar(df_corr_new.index, df_corr_new['plated'])

ax.set_xticklabels(df_corr_new.index, rotation=90)

ax.set_ylabel('correlation coefficent with plate discipline')

plt.show()



#---------------------------------------------------------------------------
#plot a scatter plot between plate discipline and per game war
plt.plot(df_new['plated'], df_new['per_war'], 'o', alpha=0.2)

#plot the linear regression line 
res = linregress(df_new['plated'], df_new['per_war'])
fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
fy = res.intercept + res.slope * fx
plt.plot(fx, fy)

plt.xlabel('plate discipline')
plt.ylabel('per_game_war')
plt.show()
#----------------------------------------------------------------------
##plot a scatter plot between plate discipline and wRC+
plt.plot(df_new['plated'], df_new['wRC+'], 'o', alpha=0.2)

#plot the linear regression line 
res = linregress(df_new['plated'], df_new['wRC+'])
fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
fy = res.intercept + res.slope * fx
plt.plot(fx, fy)

plt.xlabel('plate discipline')
plt.ylabel('wRC+')
plt.show()
#-----------------------------------------------------------------------
#plot a scatter plot between plate discipline and wOBA
plt.plot(df_new['plated'], df_new['wOBA'], 'o', alpha=0.2)

#plot the linear regression line 
res = linregress(df_new['plated'], df_new['wOBA'])
fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
fy = res.intercept + res.slope * fx
plt.plot(fx, fy)

plt.xlabel('plate discipline')
plt.ylabel('wOBA')
plt.show()
#-----------------------------------------------------------------------
#plot a scatter plot between plate discipline and ISO
plt.plot(df_new['plated'], df_new['ISO'], 'o', alpha=0.2)

#plot the linear regression line 
res = linregress(df_new['plated'], df_new['ISO'])
fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
fy = res.intercept + res.slope * fx
plt.plot(fx, fy)

plt.xlabel('plate discipline')
plt.ylabel('ISO')
plt.show()
#----------------------------------------------------------------------
#plot a scatter plot between plate discipline and OBP
plt.plot(df_new['plated'], df_new['OBP'], 'o', alpha=0.2)

#plot the linear regression line 
res = linregress(df_new['plated'], df_new['OBP'])
fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
fy = res.intercept + res.slope * fx
plt.plot(fx, fy)

plt.xlabel('plate discipline')
plt.ylabel('OBP')
plt.show()
#----------------------------------------------------------------------
#plot a scatter plot between plate discipline and AVG
plt.plot(df_new['plated'], df_new['AVG'], 'o', alpha=0.2)

#plot the linear regression line 
res = linregress(df_new['plated'], df_new['AVG'])
fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
fy = res.intercept + res.slope * fx
plt.plot(fx, fy)

plt.xlabel('plate discipline')
plt.ylabel('AVG')
plt.show()
#-----------------------------------------------------------------------
#plot a scatter plot between plate discipline and K%
plt.plot(df_new['plated'], df_new['K%'], 'o', alpha=0.2)

#plot the linear regression line 
res = linregress(df_new['plated'], df_new['K%'])
fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
fy = res.intercept + res.slope * fx
plt.plot(fx, fy)

plt.xlabel('plate discipline')
plt.ylabel('K%')
plt.show()
#-----------------------------------------------------------------------
#plot a scatter plot between plate discipline and BB%
plt.plot(df_new['plated'], df_new['BB%'], 'o', alpha=0.2)

#plot the linear regression line 
res = linregress(df_new['plated'], df_new['BB%'])
fx = np.array([df_new['plated'].min(), df_new['plated'].max()])
fy = res.intercept + res.slope * fx
plt.plot(fx, fy)

plt.xlabel('plate discipline')
plt.ylabel('BB%')
plt.show()
