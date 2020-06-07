# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:30:31 2020

@author: Nik
"""

import pandas as pd
cococola = pd.read_excel("C:\\Users\\Nik\\Downloads\\CocaCola_Sales_Rawdata.xlsx")
import numpy as np
quarter=['Q1','Q2','Q3','Q4']
n=cococola['Quarter'][0]
n[0:2]

cococola['quarter']=0

for i in range(42):
    n=cococola['Quarter'][i]
    cococola['quarter'][i]=n[0:2]
    dummy=pd.DataFrame(pd.get_dummies(cococola['quarter']))
    
coco=pd.concat((cococola,dummy),axis=1)
coco.head()
t= np.arange(1,43)
coco['t']=t
coco['t_square']=coco['t']*coco['t']


log_Sales=np.log(coco['Sales'])
coco['log_Sales']=log_Sales
     
train= coco.head(38)
test=coco.tail(4)
coco.Sales.plot()

import statsmodels.formula.api as smf

#linear model
linear= smf.ols('Sales~t',data=train).fit()
predlin=pd.Series(linear.predict(pd.DataFrame(test['t'])))
rmselin=np.sqrt((np.mean(np.array(test['Sales'])-np.array(predlin))**2))
rmselin
#421.17878760022813

#quadratic model
quad=smf.ols('Sales~t+t_square',data=train).fit()
predquad=pd.Series(quad.predict(pd.DataFrame(test[['t','t_square']])))
rmsequad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predquad))**2))
rmsequad
#475.56183518315095
#exponential model
expo=smf.ols('log_Sales~t',data=train).fit()
predexp=pd.Series(expo.predict(pd.DataFrame(test['t'])))
predexp
rmseexpo=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predexp)))**2))
rmseexpo
#466.24797310672346

#additive seasonality
additive= smf.ols('Sales~ Q1+Q2+Q3+Q4',data=train).fit()
predadd=pd.Series(additive.predict(pd.DataFrame(test[['Q1','Q2','Q3','Q4']])))
predadd
rmseadd=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predadd))**2))
rmseadd
#1860.0238154547283

#additive seasonality with linear trend
addlinear= smf.ols('Sales~t+Q1+Q2+Q3+Q4',data=train).fit()
predaddlinear=pd.Series(addlinear.predict(pd.DataFrame(test[['t','Q1','Q2','Q3','Q4']])))
predaddlinear


rmseaddlinear=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predaddlinear))**2))
rmseaddlinear
# 464.98290239822427

#additive seasonality with quadratic trend
addquad=smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=train).fit()
predaddquad=pd.Series(addquad.predict(pd.DataFrame(test[['t','t_square','Q1','Q2','Q3','Q4']])))
rmseaddquad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predaddquad))**2))
rmseaddquad
#301.73800719352977


#multiplicative seasonality
mulsea=smf.ols('log_Sales~Q1+Q2+Q3+Q4',data=train).fit()
predmul= pd.Series(mulsea.predict(pd.DataFrame(test[['Q1','Q2','Q3','Q4']])))
rmsemul= np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmul)))**2))
rmsemul
#1963.3896400779709


#multiplicative seasonality with linear trend
mullin= smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data=train).fit()
predmullin= pd.Series(mullin.predict(pd.DataFrame(test[['t','Q1','Q2','Q3','Q4']])))
rmsemulin=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmullin)))**2))
rmsemulin
#225.5243904982721

#multiplicative seasonality with quadratic trend
mul_quad= smf.ols('log_Sales~t+t_square+Q1+Q2+Q3+Q4',data=train).fit()
pred_mul_quad= pd.Series(mul_quad.predict(test[['t','t_square','Q1','Q2','Q3','Q4']]))
rmse_mul_quad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_mul_quad)))**2))
rmse_mul_quad
#581.8457187971785

#tabulating the rmse values

data={'Model':pd.Series(['rmse_mul_quad','rmseadd','rmseaddlinear','rmseaddquad','rmseexpo','rmselin','rmsemul','rmsemulin','rmsequad']),'Values':pd.Series([rmse_mul_quad,rmseadd,rmseaddlinear,rmseaddquad,rmseexpo,rmselin,rmsemul,rmsemulin,rmsequad])}
data

Rmse=pd.DataFrame(data)
Rmse

#final model with least rmse value
#coca_pred = pd.read_excel("C:\\Users\\Hello\\Desktop\\Data science\\data science\\assignments\\forcasting\\assignment data set\\CocaColapred.xl")

final= smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data=train).fit()
pred_new  = pd.Series(Mul_Add_sea.predict(test))
pred_new1=np.exp(pred_new)
pred_new1
#50    1061.452817
#51    1256.558059
#52    1450.010690
#53    1596.679670
#54    1649.807263
#55    1747.515697
#56    1766.807368
#57    1716.235167
#58    1442.176743
#59    1220.845250
test["forecasted_Sales"] = pd.Series(pred_new1)
test

#
#Out[1073]: 
#   Quarter   Sales quarter  Q1  ...   t  t_square  log_Sales  
#38   Q3_95  4895.0      Q3   0  ...  39      1521   8.495970               
#39   Q4_95  4333.0      Q4   0  ...  40      1600   8.374015               
#40   Q1_96  4194.0      Q1   1  ...  41      1681   8.341410               
#41   Q2_96  5253.0      Q2   0  ...  42      1764   8.566555              