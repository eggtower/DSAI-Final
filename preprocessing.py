import numpy as np
import pandas as pd
import pickle
import gc
import matplotlib.pyplot as plt
from itertools import product
from sklearn.preprocessing import LabelEncoder

# load data
sales = pd.read_csv('input/sales_train.csv', parse_dates=['date'], dtype={'date': 'str', 'date_block_num': 'int32', 'shop_id': 'int32', 'item_id': 'int32', 'item_price': 'float32', 'item_cnt_day': 'int32'})
items = pd.read_csv('input/items.csv', dtype={'item_name': 'str', 'item_id': 'int32', 'item_category_id': 'int32'})
shops = pd.read_csv('input/shops.csv', dtype={'shop_name': 'str', 'shop_id': 'int32'})
cats = pd.read_csv('input/item_categories.csv', dtype={'item_category_name': 'str', 'item_category_id': 'int32'})
test = pd.read_csv('input/test.csv', dtype={'ID': 'int32', 'shop_id': 'int32', 'item_id': 'int32'})
submission = pd.read_csv('input/sample_submission.csv')

cols = ['date_block_num','shop_id','item_id']

# 移除離群值
sales = sales[(sales['item_price'] < 50000 )& (sales['item_cnt_day'] < 750) & (sales['item_price'] > 0) & (sales['item_cnt_day'] > 0)]

# 將相同店名的商店設為同類
sales['shop_id']=sales['shop_id'].replace({0:57,1:58,11:10,40:39,23:24})
sales=sales[~sales['shop_id'].isin([8,9,20,23,32])]
test['shop_id']=test['shop_id'].replace({0:57,1:58,11:10,40:39,23:24}) 
sales.reset_index(drop = True,inplace=True)
shops.loc[ shops['shop_name'] == 'Сергиев Посад ТЦ "7Я"',"shop_name" ] = 'СергиевПосад ТЦ "7Я"'

# 每個商店位於的都市
shops["city"] = shops['shop_name'].str.split(" ").map( lambda x: x[0] )
shops.loc[shops["city"] == "!Якутск", "city"] = "Якутск"
shops["shop_city"] = LabelEncoder().fit_transform( shops["city"] )

# 商店歸類
shops["category"] = shops['shop_name'].str.split(" ").map( lambda x: x[1] )
shop_location_dict={'ТК':1,'ТЦ':4,'ТРК':2,'ТРЦ':3,'МТРЦ':0}
shops['shop_category'] = shops['category'].apply( lambda x: shop_location_dict[x] if x in shop_location_dict else 0 )

shops = shops[["shop_id", "shop_category", "shop_city"]]

# 每個商品類別
cats["type"] = cats['item_category_name'].apply( lambda x: x.split(" ")[0] ).astype(str)
category = []
for cat in cats["type"].unique():
    if len(cats[cats["type"] == cat]) >= 5: 
        category.append( cat )
cats["subtype"] = cats["type"].apply(lambda x: x if (x in category) else "others")

cats['main_type'] = LabelEncoder().fit_transform(cats['type'])
cats['sub_type'] = LabelEncoder().fit_transform(cats['subtype'])
cats = cats[['item_category_id','main_type', 'sub_type']]

items.drop(['item_name'], axis=1, inplace=True)

# Monthly sales
matrix = []
for i in range(34):
    sale = sales[sales['date_block_num']==i]
    matrix.append(np.array(list(product([i], sale['shop_id'].unique(), sale['item_id'].unique())), dtype='int16'))
    
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols,inplace=True)

# 計數商品每個月的銷售量
sales['revenue'] = sales['item_price'] *  sales['item_cnt_day']
group = sales.groupby(cols).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)
# 將train轉成(0,20)
matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month'].fillna(0).clip(0,20).astype(np.float32))

#Test set
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)

matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) # 34 month

# Merge Shops/Items/Cats features
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
matrix['shop_city'] = matrix['shop_city'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['main_type'] = matrix['main_type'].astype(np.int8)
matrix['sub_type'] = matrix['sub_type'].astype(np.int8)
matrix['month'] = matrix['date_block_num'] % 12
matrix['year'] = (matrix['date_block_num'] / 12).astype(np.int8)

# Month since last sale for each shop/item pair.
last_sale = pd.DataFrame()
for month in range(1,35):    
    last_month = matrix.loc[(matrix['date_block_num']<month)&(matrix['item_cnt_month']>0)].groupby(['item_id','shop_id'])['date_block_num'].max()
    df = pd.DataFrame({
        'date_block_num':np.ones([last_month.shape[0],])*month,
        'item_id': last_month.index.get_level_values(0).values,
        'shop_id': last_month.index.get_level_values(1).values,
        'item_shop_last_sale': last_month.values
        })
    last_sale = last_sale.append(df)
last_sale['date_block_num'] = last_sale['date_block_num'].astype(np.int8)

matrix = matrix.merge(last_sale, on=['date_block_num','item_id','shop_id'], how='left')

# Month since last sale for each item.
last_sale = pd.DataFrame()
for month in range(1,35):    
    last_month = matrix.loc[(matrix['date_block_num']<month)&(matrix['item_cnt_month']>0)].groupby('item_id')['date_block_num'].max()
    df = pd.DataFrame({
        'date_block_num':np.ones([last_month.shape[0],])*month,
        'item_id': last_month.index.values,
        'item_last_sale': last_month.values
        })
    last_sale = last_sale.append(df)
last_sale['date_block_num'] = last_sale['date_block_num'].astype(np.int8)

matrix = matrix.merge(last_sale, on=['date_block_num','item_id'], how='left')

# Months since the first sale for each shop/item pair and for item only

matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')

matrix = matrix[matrix.date_block_num > 11]
# matrix.columns
print(matrix.columns)
matrix.to_pickle('data.pkl')
del matrix
del group
del items
del shops
del cats
del sales
# leave test for submission
gc.collect()
print('......Done....')