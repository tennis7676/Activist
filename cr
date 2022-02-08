import pandas as pd
Index = pd.read_csv('C:/Users/kosuke/ファクター分析/Data/MarketIndex_new.csv')
Index

for s in symbol_list: 
    ####
    df2 = pd.DataFrame(requests.get(url).json())
    df2 = df2.rename(columns={0:"Time", 4:"Close"})
    df2["Symbol"] = s
    df2 = df2[["Symbol", "Time", "Close"]]
        
    ###

df = df.sort_values(["Time","Symbol"]).reset_index(drop=True)
df = df[df["Time"] != df["Time"][len(df)-1]]    #一番最後の日付はまだ確定していない
df = df.sort_values(by = ["Symbol", "Time"]).reset_index(drop=True)    #これ大事
df = df.astype({"Close":"float64"})
df["Ret"] = df.groupby("Symbol")["Close"].pct_change(1)
df = df.dropna()
df = df.reset_index(drop=True)
df

#クロスセクションで異常値処理
df = df.sort_values(by = ["Time", "Symbol"]).reset_index(drop=True)
df["RetMean"] = df.groupby("Time")["Ret"].transform("mean")
df["RetStd"] = df.groupby("Time")["Ret"].transform("std", ddof=1)
#df["Ret"] = np.where(df["Ret"] < df["RetMean"] - 3 * df["RetStd"], df["RetMean"] - 3 * df["RetStd"], df["Ret"])
#df["Ret"] = np.where(df["Ret"] > df["RetMean"] + 3 * df["RetStd"], df["RetMean"] + 3 * df["RetStd"], df["Ret"])
df["Ret"] = np.where(df["Ret"] < df["RetMean"] - 2 * df["RetStd"], df["RetMean"] - 2 * df["RetStd"], df["Ret"])
df["Ret"] = np.where(df["Ret"] > df["RetMean"] + 2 * df["RetStd"], df["RetMean"] + 2 * df["RetStd"], df["Ret"])
#df["flag"] = np.where(df["Ret"] != df["AdjRet"], 1, 0)
df = df.sort_values(by = ["Symbol", "Time"]).reset_index(drop=True)
df

#β計算
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

window = 90

for s in symbol_list:
    df_one = df[df["Symbol"] == s].copy()
    
    if len(df_one) >= window:
        df_one = pd.merge(df_one, Index, how="inner", on="Time")
        df_one = df_one.sort_values(by = "Time").reset_index(drop=True)
    
        for i in range(window - 1, len(df_one)):
            x = df_one.loc[i - window + 1:i, "IndexRet"]
            y = df_one.loc[i - window + 1:i, "Ret"]
            model_lr = LinearRegression()
            model_lr.fit(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
            df_one.loc[i, "beta"] = model_lr.coef_[0][0]

        df_one = df_one[["Time", "Symbol", "beta"]]
        if s == symbol_list[0]:
            df_beta = df_one
        else:
            df_beta = pd.concat([df_beta, df_one], axis=0, join="inner")
    
df_beta = df_beta.dropna()
df_beta = df_beta.sort_values(by = ["Symbol", "Time"]).reset_index(drop=True)

#pd.set_option('display.max_rows', None)
df_beta
