Introduction:

Try to see if plate discipline in baseball affected other stats including regular ones like BB%, K%, AVG, OBP, SLG, OPS, ISO, wOBA, wRC+, etc

Also see if plate discipline in baseball affected other stats like WAR per game to see how plate discipline affect winning


Methods:

Data was from FanGraphs

Use SQL Server and Python(Spyder)

Use linear regression 

Plate Discipline defined as (ZSwing%-OSwing%)/Swing%

Find out the correlation coefficient, linear regression line, ratio and heritability, all of them were performed with pairs bootstrap as well, in order to analyze probabilistically and get the confidence interval.


Results:

For results of relationship between plate discipline(plated) and per game war(pgw):

correlation coefficient = 0.274454416886356 [0.19710456 0.3450419 ]

linear regression slope = 0.005104575299609698 conf int = [0.00357109 0.00659355]

linear regression intercept = -0.0007680167310351522 conf int = [-0.00200369  0.00046052]

pgw/plated mean = 0.004112620026148256 conf int = [0.00379471 0.00442882]

heritability = 0.005104575299609694 [0.00357444 0.00679131]



For results of relationship between plate discipline(plated) and BB%:

correlation coefficient = 0.8145041162207632 [0.78578042 0.8393538 ]

linear regression slope = 0.15849852437384196 conf int = [0.1489552  0.16864826]

linear regression intercept = -0.04300963219489709 conf int = [-0.05053895 -0.0359336 ]

BB%/plated mean = 0.10289894365749544 conf int = [0.100784   0.10505148]

heritability = 0.15849852437384201 [0.14896195 0.16807283]



For results of relationship between plate discipline(plated) and K%:

correlation coefficient = 0.13528870001040819 [0.05699777 0.21395654]

linear regression slope = 0.05832271453660731 conf int = [0.02501156 0.09253158]

linear regression intercept = 0.15416511596940596 conf int = [0.12737901 0.18045186]

K%/plated mean = 0.258099882637668 conf int = [0.24936942 0.26733477]

heritability = 0.05832271453660738 [0.02381265 0.09467595]


Conclusion:

Out of pgw(per game war), BB%, K%, BB% is obviously more related to plate discipline, as shown in both correlation coefficient and heritability




