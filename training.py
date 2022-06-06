import numpy as np
import pandas as pd
import argparse
from itertools import product
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class Train:
    
    def __init__(self, DataName,modelname):
        self.DataName = DataName
        self.model = None
        self.data =None
        self.cat_feats = None
        self.modelname = modelname
        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None
        self.X_test = None

    def data_loader(self):
        self.data = pd.read_pickle(self.DataName)
    #     self.data = self.data[['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'ID',
    #    'shop_category', 'shop_city', 'item_category_id', 'main_type',
    #    'sub_type', 'sub_type2', 'month', 'year', 'item_shop_last_sale',
    #    'item_last_sale', 'item_shop_first_sale', 'item_first_sale']]
        self.cat_feats = ['shop_id','shop_city','item_category_id','main_type','sub_type']

    def dataset_split(self,data):
        self.X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
        self.Y_train = data[data.date_block_num < 33]['item_cnt_month']
        self.X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
        self.Y_valid = data[data.date_block_num == 33]['item_cnt_month']
        self.X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

    def XGB_model_building(self):
        model = XGBRegressor(
            booster ='gbtree',
            objective = 'reg:squarederror',
            eval_metric='rmse',
            gamma = 3.866115369278931e-08,
            min_child_weight = 36.28309377320559,
            max_depth = 11,
            subsample = 0.5871817395511052,
            colsample_bytree= 0.8278290725391293,
            tree_method = 'exact',
            learning_rate = 0.01,
            n_estimators = 2000,
            seed = 42)

        model.fit(
            self.X_train, 
            self.Y_train, 
            eval_metric="rmse", 
            eval_set=[(self.X_train, self.Y_train), (self.X_valid, self.Y_valid)], 
            verbose=50, 
            early_stopping_rounds = 20,
            )
        return model
    
    def LGBM_model_building(self):
        model = LGBMRegressor(
            bagging_fraction = 0.7,
            boosting_type = 'gbdt',
            colsample_bytree = 0.7,
            feature_fraction = 0.7,
            learning_rate = 0.1,
            min_child_weight = 100,
            max_depth = 8,
            n_estimators = 800,
            num_leaves = 1000,
            reg_alpha = 120,
            reg_lambda = 1.1,
            random_state = 42)

        model.fit(
            self.X_train, 
            self.Y_train, 
            eval_metric="rmse", 
            eval_set=[(self.X_train, self.Y_train), (self.X_valid, self.Y_valid)], 
            verbose=10, 
            early_stopping_rounds = 40,
            categorical_feature = self.cat_feats)

        return model


    def CatBoost_model_building(self):
        model = CatBoostRegressor(
            iterations=1000, loss_function='RMSE',
            learning_rate=0.06,  
            depth=8,              
            l2_leaf_reg=11,
            random_seed=17, 
            silent=True,)


        model.fit( self.X_train, self.Y_train, 
                    cat_features=self.cat_feats,
                    early_stopping_rounds = 40,
                    verbose=10
                    )
        return model

    def test(self):
        Y_pred = self.model.predict(self.X_valid).clip(0, 20)
        Y_test = self.model.predict(self.X_test).clip(0, 20)

        X_train_level2 = pd.DataFrame({
            "ID": np.arange(Y_pred.shape[0]), 
            "item_cnt_month": Y_pred
        })

        submission = pd.DataFrame({
            "ID": np.arange(Y_test.shape[0]), 
            "item_cnt_month": Y_test
        })
        submission.to_csv('submission.csv', index=False)

    def main(self):
        self.data_loader()
        self.dataset_split(self.data)
        if self.modelname == 'XGB':
            self.model = self.XGB_model_building()
        elif self.modelname == 'LGBM':
            self.model = self.LGBM_model_building()
        elif self.modelname == 'CatBoost':
            self.model = self.CatBoost_model_building()

        self.test()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre',
                        default='./data.pkl',
                        help='input data file name')
    
    args = parser.parse_args()
    
    Train = Train(args.pre,'XGB')
    # Train = train(args.pre,'LGBM')
    # Train = train(args.pre,'CatBoost')
    Train.main()