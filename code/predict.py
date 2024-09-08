import torch
from datetime import datetime, timedelta
import yfinance as yf
from data_preprocess import DataPreProcess
from sklearn.preprocessing import MinMaxScaler
from model import LSTM
import numpy as np
from script import send_sms
import time



def predict(current_date):

    def calculate_x_days_ago(current_date,days):
        business_days_counted = 0
        while business_days_counted <= days:
            current_date -= timedelta(days=1)
            if current_date.weekday() < 5:  
                business_days_counted += 1
        return current_date


    date_20_windows = calculate_x_days_ago(current_date,40)


    data = yf.download("EURPLN=X", start=date_20_windows.strftime("%Y-%m-%d"), end=current_date.strftime("%Y-%m-%d"))


    data_to_predict = DataPreProcess(
        df=data,
        pred_column='Close',
        windows=20,
        columns=['Open','High','Low','Close','Adj Close'],
        scaler=MinMaxScaler
    )




    X = data_to_predict.preprocess_df()

    model = LSTM(data_to_predict.n_features,6,1)

    model.load_state_dict(torch.load('polxeur_model.pth'))

    model.eval()
    with torch.inference_mode():
        preds = model(X)
        
        
        inverse_input = np.zeros((1, 101))  # Tworzy macierz z zerami
        inverse_input[0, 0] = preds.item()
        
        un_scaled_values = data_to_predict.scaler_obj.inverse_transform(inverse_input)
        real_value = un_scaled_values[0,0]
        send_sms(real_value)






sended_today = False
while True:
    current_date = datetime.now()
    
    if current_date.hour == 12 and current_date.minute== 0 and not sended_today:
        predict(current_date)
        sended_today = True
        
    if current_date.hour == 0 and current_date.minute == 0 :
        sended_today = False
        
    time.sleep(60)


