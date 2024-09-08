import pandas as pd
from copy import deepcopy as dp
import torch

class DataPreProcess():
  def __init__(self,
               df:pd.DataFrame,
               pred_column:str,
               windows,
               columns,
               scaler
               ):
    self.df = dp(df)
    self.n_features = len(columns)
    self.columns = columns
    self.pred_column = pred_column
    self.scaler = scaler
    self.scaler_obj = None
    self.windows = windows
    self.prepare_df()
    self.shift_data()

  def prepare_df(self):
    
    temp = self.df.iloc[len(self.df)-1]
    self.df.loc[len(self.df.index)] = temp

    self.df = self.df[self.columns]

  def shift_data(self):

    features = self.columns

    shiffted_df = []
    for i in range(1,self.windows+1):
      shiffted = self.df[features].shift(i)
      shiffted.columns = [f"{feature} n-{i}" for feature in features]
      shiffted_df.append(shiffted)


    features.remove(self.pred_column)
    self.df.drop(columns=features,inplace=True)
    
    df_s = pd.concat([self.df]+shiffted_df,axis=1)
    df_s.dropna(inplace=True)
    df_s.columns = [df_s.columns[0]] + df_s.columns[1:][::-1].tolist()
    df_s.iloc[:,1:] = df_s.iloc[:,:0:-1]
    self.df = df_s


  def preprocess_df(self):
  
    df_n = self.df.to_numpy()

    self.scaler_obj = self.scaler()
    df_n_scaled = self.scaler_obj.fit_transform(df_n)
        
    X = torch.tensor(df_n_scaled[-1:,1:]).float()

    X = X.reshape((-1,self.windows,self.n_features))
    
    return X