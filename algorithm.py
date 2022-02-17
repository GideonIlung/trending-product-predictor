#IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import re
from statsmodels.tsa.stattools import adfuller #used to check stationarity of data#
from statsmodels.tsa.arima.model import ARIMA #arima model used for time series analysis#
from pmdarima import auto_arima #optimises the arima parameters for a specific dataset#

#algorithm written in python script#
main_data = pd.read_csv('sales_volumes.csv')
baseline = pd.read_csv('baseline.csv')
main_data['date'] = pd.to_datetime(main_data['date'])
main_data['volume'] = main_data['volume'].astype('float64')
start_date = datetime(2019,12,31)
end_date = datetime(2020,6,29)

product_list = main_data['description'].dropna().unique()
output = pd.DataFrame(columns=['product_code','baseline','predicted','trending'])

for product_name in product_list:
    sub_data = main_data[main_data['description']==product_name]
    product_code = sub_data['product_code'].unique()[0]
    sub_data = main_data[main_data['description']==product_name]
    formated_data = pd.DataFrame(columns=['date','volume'])

    #getting volume sales for the day,including missing dates#
    curr_date = start_date

    while curr_date <= end_date:
        x = sub_data[sub_data['date']==curr_date]
        vol = 0

        if x.shape[0]>0:
            vol = sum(x['volume'])

        row = {'date':curr_date,'volume':vol}
        formated_data = formated_data.append(row,ignore_index=True)
        curr_date+=timedelta(days=1)

    formated_data.set_index('date',inplace=True)
    formated_data.index = pd.DatetimeIndex(formated_data.index.values,freq=formated_data.index.inferred_freq)

    #replacing outliers using interpolation#
    formated_data['Z_score'] = (formated_data['volume'] - formated_data['volume'].mean())/formated_data['volume'].std()
    formated_data['volume'].loc[np.abs(formated_data['Z_score'] > 3)] = np.nan
    formated_data['volume'].interpolate(method='polynomial',order=2,axis=0,inplace=True)
    formated_data = formated_data.drop(['Z_score'],axis=1)
    formated_data = formated_data.fillna(value=formated_data.mean())
    
    #fitting the arima model and getting predictions#
    param_fit = auto_arima(formated_data,trace=False,suppress_warnings=True)
    p,d,q=param_fit.get_params()['order']

    if p+d+q == 0: #if optimal parameters cannot be found use default values (1,1,1)#
        p = 1
        d = 1
        q = 1

    model = ARIMA(formated_data.astype(float),order=(p,d,q))
    model = model.fit()

    start = formated_data.index[-1] + timedelta(days=1)
    end = start + timedelta(days=99)
    pred = model.predict(start=start,end=end)

    #getting total sales for next 4 weeks#
    pred_data = pd.DataFrame(pred)
    end = start + timedelta(weeks=4)
    predicted_volume = pred_data[pred_data.index<=end].sum()[0]
    
    base_data = baseline['baseline_forecast'].loc[baseline['product_code']==product_code]
    base_volume = 0

    #getting baseline volume#
    if base_data.shape[0] ==1:
        base_volume = base_data.values[0]
    
    #determining if product trending#
    trending = False

    if base_volume!=0:
        percent = ((predicted_volume-base_volume)/base_volume)*100
        if percent>=100:
            trending=True
        else:
            trending=False
    
    record = {'product_code':product_code,'baseline':base_volume,'predicted':predicted_volume,'trending':trending}
    output = output.append(record,ignore_index=True)

output.set_index('product_code',inplace=True)
output.to_csv('output.csv')