import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from arch import time

pd.set_option('display.max_columns', 999)

def confusion_matrix(y_true, y_pred):
    print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

raw_df = pd.read_csv('featured_data.csv', index_col='datetime', parse_dates=True)
print(raw_df.describe())

excluded_list = ['model_encoded', 'machineID', 'error1', 'error2', 'error3', 'error4','error5', 'comp1', 'comp2', 'comp3', 'comp4']
label_list = ['failure_comp1', 'failure_comp2', 'failure_comp3', 'failure_comp4']
s = set(excluded_list + label_list)
print('exclude columns:', set(excluded_list + label_list))

feature_list = [x for x in raw_df.columns if x not in set(excluded_list + label_list)]
print('feature columns:', feature_list)

print('sainty check')
_,n = raw_df.shape
print("{} columns in raw data,\n{} columns for feature".format(n, len(feature_list)))

# most of rolling value are zeros at beginning
df_train = raw_df[(raw_df.index <= '2015-10-30') & (raw_df.index >= '2015-02-01')]
# failure label is not accurate, becuase we don't have information after 2016-01-01
df_test = raw_df[(raw_df.index > '2015-10-30') & (raw_df.index <= '2015-12-10')]
x_train = df_train[feature_list].get_values()
y_train = df_train[label_list].get_values()
x_test = df_test[feature_list].get_values()
y_test = df_test[label_list].get_values()
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test  shape:', x_test.shape)
print('y_test  shape:', y_test.shape)


scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



for i in range(4):
    clf = RandomForestClassifier(n_estimators=200,
                                 criterion='gini',
                                 max_depth=15,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 max_features='sqrt',
                                 n_jobs=4,
                                 verbose=1)

    start_time = time.time()

    clf.fit(x_train, y_train[:, i])

    elapsed_time = time.time() - start_time
    print("\nComplete", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


    y_pred = clf.predict(x_test)
    print(clf.score(x_test, y_test[:, i]))
    print("Confusion Matrix")
    print(confusion_matrix(y_true=y_test[:, i], y_pred=y_pred))

