# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:10:17 2020

@author: Nik
"""
import pandas as pd
import numpy as np
%matplotlib inline
# Load specific forecasting tools
from statsmodels.tsa.ar_model import AR,ARResults

air = pd.read_excel("C:\\Users\\Nik\\Downloads\\Airlines+Data.xlsx")
air.index.freq = 'MS'

import warnings
warnings.filterwarnings("ignore")

air.shape
month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
month=month*8
month1=pd.DataFrame(month)
month1.shape



air['months']= month1
air.head()


#Air['Month'] = pd.to_datetime(Air['Month'], infer_datetime_format=True)

#n = air['Month'][0]
#n[0:3]



    
month_dummies = pd.DataFrame(pd.get_dummies(air['months']))
air1 = pd.concat([air,month_dummies],axis = 1)

air1["t"] = np.arange(1,97)
   

air1["t_squared"] = air1["t"]*air1["t"]
air1.columns
air1["log_Passengers"] = np.log(air1["Passengers"])
#air1.rename(columns={"Ridership ('000)": 'Ridership'}, inplace=True)
air1.Passengers.plot()
Train = air1.head(90)
Test = air1.tail(7)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear
#64.94990126633445
##################### Exponential ##############################

Exp = smf.ols('log_Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
#58.68851328548133
#################### Quadratic ###############################

Quad = smf.ols('Passengers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad
#59.370475445546845
################### Additive seasonality ########################

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea
#139.28817632570227
################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
# 30.17322302715035
################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
# 11.8416290599352
##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea #178.97551501505967
#11.8416290599352
################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
#               MODEL  RMSE_Values
#0        rmse_linear    64.949901
#1           rmse_Exp    58.688513
#2          rmse_Quad    59.370475
#3       rmse_add_sea   139.288176
#4  rmse_add_sea_quad    30.173223
#5      rmse_Mult_sea    11.841629
#6  rmse_Mult_add_sea    11.841629
# so rmse_add_sea has the least value among the models prepared so far 
# Predicting new values 

#predict_data = pd.read_csv("C:\\Users\\Nik\\Downloads\\Predict_new.xlsx")
model_full = smf.ols('log_Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()

pred_new  = pd.Series(Mul_Add_sea.predict(Test))
pred_new1=np.exp(pred_new)
pred_new1
Out[1169]: 
#89    355.358491
#90    392.715783
#91    392.863241
#92    352.940154
#93    310.697077
#94    272.349052
#95    312.869324
Test["forecasted_Passengers"] = pd.Series(pred_new1)
test
#out
#Quarter   Sales quarter  Q1  ...   t  t_square  log_Sales  
#38   Q3_95  4895.0      Q3   0  ...  39      1521   8.495970               
#39   Q4_95  4333.0      Q4   0  ...  40      1600   8.374015               
#40   Q1_96  4194.0      Q1   1  ...  41      1681   8.341410               
#41   Q2_96  5253.0      Q2   0  ...  42      1764   8.566555