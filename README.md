Use data from FanGraphs/

Use SQL Server and Python(Spyder)/

Use linear regression to see if plate discipline in baseball affected other stats including regular ones like BB%, K%, AVG, OBP, SLG, OPS, ISO, wOBA, wRC+, etc/

Also see if plate discipline in baseball affected other stats like WAR per game to see how plate discipline affect winning/

Plate Discipline defined as (ZSwing%-OSwing%)/Swing%

Find out the correlation coefficient, linear regression line, ratio and heritability, all of them were performed with pairs bootstrap as well, in order to analyze probabilistically and get the confidence interval of all of them.


For results of relationship between plate discipline(plated) and per game war(pgw):
correlation coefficient = 0.274454416886356 [0.19710456 0.3450419 ]
linear regression slope = 0.005104575299609698 conf int = [0.00357109 0.00659355]
linear regression intercept = -0.0007680167310351522 conf int = [-0.00200369  0.00046052]
pgw/plated mean = 0.004112620026148256 conf int = [0.00379471 0.00442882]
heritability = 0.005104575299609694 [0.00357444 0.00679131], p-val = 1.0

Imply that plate discipline is not highly related to per game war, and there is near to no heritability from plate discipline to per game war as well.
