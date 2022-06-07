# DSAI-Final

## 使用說明

環境 `python:3.7.13`
安裝相依套件
```
pip install -r requirements.txt
```
資料前處理
```
python featureEngineering.py
```
模型訓練
```
python training.py
```

## 檔案說明

- `preprocessing.py`：進行資料前處理的檔案，會將 training data 製作成特徵，儲存在 `data.pkl`。
- `training.py`：模型訓練與競賽預測，預設模型為 XGBoost，可以透過修改主程式輸入的模型代號更換訓練的模型(`LGBM`,`CatBoost`)
- `data.pkl`：保存特徵的檔案。
- `submission.csv`：經由運行`training.py`產生的檔案，競賽最終所繳交的檔案。
- `input` dir：
  -  `sales_train.csv` ：主要的訓練資料。2013 年 1 月至 2015 年 10 月的每日銷售歷史數據
  -  `test.csv`：測試集。需要預測商店和產品在 2015 年 11 月的銷售額
  -  `sample_submission.csv`：提交文件的正確格式
  -  `items.csv` :有關商品/產品的補充資訊
  -  `item_categories.csv`：有關商品類別的補充資訊
  -  `shops.csv`：有關商店的補充資訊

![image](https://user-images.githubusercontent.com/13596525/172260736-c0c621a9-1e0b-4bb3-a416-243ed0a3a569.png)
![image](https://user-images.githubusercontent.com/13596525/172260755-6149f9d3-6e30-43fa-ae17-ad1356113fb3.png)

## 模型選擇

採用Kaggle競賽常用的模型，包含`XGBoost`、`LightGBM`、`CatBoost`。

## 訓練結果

[競賽](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/overview) 採用 RMSE 來做為評分標準

||XGBoost|LightGBM|CatBoost|
|--|--|--|--|
|Score|0.99607|1.01463|1.01070|
