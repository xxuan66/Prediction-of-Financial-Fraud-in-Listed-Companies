import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor, \
    GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import linear_model
import lightgbm as lgb
import xgboost as xgb
from collections import Counter
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore")


# # 数据读取及预处理
def load_data():
    file_1 = pd.read_excel('附件1.xlsx')
    file_2 = pd.read_csv('附件2.csv')
    file_3 = pd.read_excel('附件3.xlsx')
    return file_1, file_2, file_3


# 数据合并及分行业处理
def preprocess_by_industry(file_1, file_2):
    file_1.columns = ['TICKER_SYMBOL', '所属行业']
    data = pd.merge(file_1, file_2, how='right', on='TICKER_SYMBOL')
    columns_list = data['所属行业'].value_counts().index

    for industry in columns_list:
        data_one = data[data['所属行业'] == industry]
        print(f"Processing {industry}, shape: {data_one.shape}")

        flag = data_one.pop('FLAG')

        # 清除缺失值大于50%的列
        for col in data_one.columns:
            if ((data_one[col].isnull().sum() / data_one.shape[0]) >= 0.5):
                data_one.drop(columns=[col], inplace=True)

        # 缺失值20%-50%的用均值填充
        for column in data_one.columns[(data_one.isnull().sum() / data_one.shape[0]) > 0.2]:
            mean_val = data_one[column].mean()
            data_one[column].fillna(mean_val, inplace=True)

        # 使用随机森林填充缺失值
        data_one = fill_missing_values_optimized(data_one)

        # 最大最小归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_one.iloc[:, 11:] = scaler.fit_transform(data_one.iloc[:, 11:])  # 仅归一化数值型字段

        # 合并标签列
        data_one['FLAG'] = flag

        # 保存数据
        data_one.to_csv(f'./清理后的数据/{industry}.csv', index=False)
        print(f"{industry} 完成")
        print('------------------------------------------------')


# 随机森林填充缺失值
def fill_missing_values_optimized(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_cols = df.isnull().sum()[df.isnull().sum() > 0].index

    while len(missing_cols) > 0:
        col_name = df[missing_cols].isnull().sum().idxmin()

        if col_name not in numeric_cols:
            missing_cols = missing_cols.drop(col_name)
            continue

        relevant_cols = list(set(numeric_cols) - set([col_name]))
        RF_train = df[df[col_name].notnull()][relevant_cols + [col_name]]
        RF_test = df[df[col_name].isnull()][relevant_cols]

        X_train = RF_train.drop(columns=[col_name]).fillna(0)
        y_train = RF_train[col_name]
        X_test = RF_test.fillna(0)

        model = RandomForestRegressor(random_state=1, n_jobs=-1)
        model.fit(X_train, y_train)

        df.loc[df[col_name].isnull(), col_name] = model.predict(X_test)
        missing_cols = missing_cols.drop(col_name)
        print(f"已填充 {col_name}")

    return df


# 样本不平衡处理
def handle_imbalance(X_tr, Y_tr):
    print('原始数据 y_train 分类情况：', Counter(Y_tr))

    ros = RandomOverSampler(random_state=0)
    X_ros, y_ros = ros.fit_resample(X_tr, Y_tr)
    print('随机过采样后 y_train 分类情况：', Counter(y_ros))

    sos = SMOTE(random_state=0)
    X_sos, y_sos = sos.fit_resample(X_tr, Y_tr)
    print('SMOTE采样后 y_train 分类情况：', Counter(y_sos))

    rus = RandomUnderSampler(random_state=0)
    X_rus, y_rus = rus.fit_resample(X_tr, Y_tr)
    print('随机欠采样后 y_train 分类情况：', Counter(y_rus))

    kos = SMOTETomek(random_state=0)
    X_kos, y_kos = kos.fit_resample(X_tr, Y_tr)
    print('综合采样后 y_train 分类情况：', Counter(y_kos))

    return X_ros, y_ros, X_sos, y_sos, X_rus, y_rus, X_kos, y_kos


# 模型训练与评估
def train_and_evaluate(X_kos, y_kos, X_te, Y_te):
    # AdaBoost 调参
    clf_ada = AdaBoostClassifier(random_state=2)
    ada_param_grid = {
        'n_estimators': [50, 100, 500],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    clf_ada = GridSearchCV(clf_ada, ada_param_grid, cv=3, n_jobs=-1)
    clf_ada.fit(X_kos, y_kos)
    print(f'Adaboost最佳参数: {clf_ada.best_params_}')
    pre_ada = clf_ada.predict(X_te)
    acc_te_ada = sum(pre_ada == Y_te) / len(pre_ada)
    print('Adaboost测试集准确率:', acc_te_ada)

    # RandomForest 调参
    clf_rf = RandomForestClassifier(random_state=2)
    rf_param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 4, 10],
        'min_samples_split': [2, 5, 10]
    }
    clf_rf = GridSearchCV(clf_rf, rf_param_grid, cv=3, n_jobs=-1)
    clf_rf.fit(X_kos, y_kos)
    print(f'RandomForest最佳参数: {clf_rf.best_params_}')
    pre_rf = clf_rf.predict(X_te)
    acc_te_rf = sum(pre_rf == Y_te) / len(pre_rf)
    print('RandomForest测试集准确率:', acc_te_rf)

    # LightGBM 调参
    lgb_train = lgb.Dataset(X_kos, y_kos)
    lgb_eval = lgb.Dataset(X_te, Y_te, reference=lgb_train)

    params = {
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
        'metric': [{'l2', 'auc'}],
        'num_leaves': [31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'feature_fraction': [0.8, 0.9],
        'bagging_fraction': [0.8, 0.9],
        'bagging_freq': [5, 10],
        'verbose': [1]
    }

    gbm = GridSearchCV(lgb.LGBMClassifier(), params, cv=3, n_jobs=-1)
    gbm.fit(X_kos, y_kos)
    print(f'LightGBM最佳参数: {gbm.best_params_}')
    y_pred = gbm.predict(X_te)
    acc_te_lgb = sum((y_pred > 0.5) == Y_te) / len(Y_te)
    print('LightGBM测试集准确率:', acc_te_lgb)

    # XGBoost 调参
    clf_xgb = xgb.XGBClassifier(random_state=2)
    xgb_param_grid = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 6, 10],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    clf_xgb = GridSearchCV(clf_xgb, xgb_param_grid, cv=3, n_jobs=-1)
    clf_xgb.fit(X_kos, y_kos)
    print(f'XGBoost最佳参数: {clf_xgb.best_params_}')
    pre_xgb = clf_xgb.predict(X_te)
    acc_te_xgb = sum(pre_xgb == Y_te) / len(pre_xgb)
    print('XGBoost测试集准确率:', acc_te_xgb)

    return clf_ada, clf_rf, gbm, clf_xgb


# 特征重要性可视化
def plot_feature_importance(clf, data_label, model_name):
    if isinstance(clf, lgb.Booster) or isinstance(clf, GridSearchCV):
        importances = clf.best_estimator_.feature_importances_
    else:
        importances = clf.feature_importances_

    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(20, 8))
    plt.title(f"{model_name} Feature Importances")
    plt.barh(range(30), importances[indices][:30], height=0.7, color='steelblue', alpha=0.8)
    plt.yticks(range(30), data_label.columns.values[:][indices][:30])
    plt.show()


# 提取重要特征
def extract_important_features(clf_ada, clf_rf, gbm, clf_xgb, data_label):
    lgb_features = data_label.columns[
        np.argsort(gbm.best_estimator_.feature_importances_)[::-1][:40]].tolist() if isinstance(gbm,
                                                                                                GridSearchCV) else []
    adaboost_features = data_label.columns[np.argsort(clf_ada.best_estimator_.feature_importances_)[::-1][:40]].tolist()
    rf_features = data_label.columns[np.argsort(clf_rf.best_estimator_.feature_importances_)[::-1][:40]].tolist()
    xgb_features = data_label.columns[np.argsort(clf_xgb.best_estimator_.feature_importances_)[::-1][:40]].tolist()

    common_features = list(set(adaboost_features) & set(rf_features) & set(lgb_features) & set(xgb_features))
    print('Adaboost提取的特征：', adaboost_features)
    print('RandomForest提取的特征：', rf_features)
    print('LightGBM提取的特征：', lgb_features)
    print('XGBoost提取的特征：', xgb_features)
    print('四个模型的共有特征：', common_features)
    return common_features


# 加载并选择特征
def load_and_select_features(common_features, industry):
    new_col = common_features.copy()
    new_col.append('TICKER_SYMBOL')
    new_col.append('FLAG')

    os.chdir('./清理后的数据/')
    data_rate = pd.read_csv(f'{industry}.csv')
    data_rate = data_rate[new_col]

    data_X = data_rate[~data_rate['FLAG'].isnull()]
    data_Y = data_rate[data_rate['FLAG'].isnull()]
    data_X['FLAG'] = data_X['FLAG'].apply(lambda x: int(x))

    return data_X, data_Y


# 构建Stacking特征
def stacking_features(model_lgb, model_xgb, model_gbdt, model_LR, x_train, x_val, x_test):
    train_lgb_pred = model_lgb.predict(x_train)
    train_xgb_pred = model_xgb.predict(x_train)
    train_gbdt_pred = model_gbdt.predict(x_train)
    train_LR_pred = model_LR.predict(x_train)

    Strak_X_train = pd.DataFrame({
        'Method_1': train_lgb_pred,
        'Method_2': train_xgb_pred,
        'Method_3': train_gbdt_pred,
        'Method_4': train_LR_pred
    })

    val_lgb_pred = model_lgb.predict(x_val)
    val_xgb_pred = model_xgb.predict(x_val)
    val_gbdt_pred = model_gbdt.predict(x_val)
    val_LR_pred = model_LR.predict(x_val)

    Strak_X_val = pd.DataFrame({
        'Method_1': val_lgb_pred,
        'Method_2': val_xgb_pred,
        'Method_3': val_gbdt_pred,
        'Method_4': val_LR_pred
    })

    test_lgb_pred = model_lgb.predict(x_test)
    test_xgb_pred = model_xgb.predict(x_test)
    test_gbdt_pred = model_gbdt.predict(x_test)
    test_LR_pred = model_LR.predict(x_test)

    Strak_X_test = pd.DataFrame({
        'Method_1': test_lgb_pred,
        'Method_2': test_xgb_pred,
        'Method_3': test_gbdt_pred,
        'Method_4': test_LR_pred
    })

    return Strak_X_train, Strak_X_val, Strak_X_test


# 模型训练与预测
def train_and_predict(data_X, data_Y, common_features):
    X_1 = data_X[data_X['FLAG'] == 1]
    X_0 = data_X[data_X['FLAG'] == 0].sample(1000)
    X_0 = X_0.append(X_1)
    X_tr, X_te, Y_tr, Y_te = train_test_split(X_0.iloc[:, :-2], X_0.iloc[:, -1], test_size=0.25)

    sos = SMOTE(random_state=0)
    X_sos, y_sos = sos.fit_resample(X_tr, Y_tr)

    model_gbdt = build_model_gbdt(X_sos, y_sos)
    model_xgb = build_model_xgb(X_sos, y_sos)
    model_lgb = build_model_lgb(X_sos, y_sos)
    model_LR = build_model_LR(X_sos, y_sos)

    Strak_X_train, Strak_X_val, Strak_X_test = stacking_features(
        model_lgb, model_xgb, model_gbdt, model_LR, X_sos, X_te, data_Y[common_features]
    )

    model_lr_Stacking = build_model_RF(Strak_X_train, y_sos)
    subA_Stacking = model_lr_Stacking.predict(Strak_X_test)

    data_Y['FLAG'] = subA_Stacking
    data_Y[['TICKER_SYMBOL', 'FLAG']].to_csv(f'stacking_{data_Y["所属行业"].iloc[0]}.csv', index=False)


# 主流程
def main():
    file_1, file_2, file_3 = load_data()

    # 分行业预处理数据
    preprocess_by_industry(file_1, file_2)

    # 逐个行业进行特征选择和模型训练
    columns_list = file_1['所属行业'].value_counts().index
    for industry in columns_list:
        # 加载并选择行业特征
        common_features = ['你的模型所需特征列表']  # 在此处放置共有特征
        data_X, data_Y = load_and_select_features(common_features, industry)

        # 执行训练与预测
        train_and_predict(data_X, data_Y, common_features)


if __name__ == "__main__":
    main()
