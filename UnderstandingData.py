import pandas as pd
import numpy as np
import csv
from sklearn.cluster import KMeans
from dateutil import parser
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer

dataset=pd.read_csv("Data/clustered_train.csv",sep=",")
testset=pd.read_csv("Data/cleanedData.csv",sep=",")
#testset=testset.sample(frac=0.001)
#kmeans = KMeans(n_clusters=100, random_state=0).fit(dataset[["pickup_latitude","pickup_longitude"]].sample(frac=0.3))
#dataset["pickup_location"]=kmeans.predict(dataset[["pickup_latitude","pickup_longitude"]])
#dataset.to_csv("Data/clustered_train.csv",sep=",")
#for i in range(0,100):
#    mean=np.mean(dataset[dataset.pickup_location==i]["fare_amount"])
#    dataset.loc[(dataset.pickup_location==i),"avg_fare"]=mean

#dataset['payment_type']=dataset['payment_type'].map({'NOC ':0,'CSH': 1, 'CRD': 2,'DIS ':3,'UNK ':4})
#dataset.loc[(dataset.tip_amount.isnull()),"tip_amount"]=0
#dataset.to_csv("Data/clustered_train.csv",sep=",")
dataset=dataset.dropna(subset=["surcharge"])

listOfColumnsInConsideration=["tolls_amount", "tip_amount", "mta_tax","surcharge","total_time","avg_fare"]

mean_pickuphour=np.mean(dataset['pickup_hour'])
mean_totaltime=np.mean(dataset['total_time'])

def copy(row):

    if row["pickup_datetime"] != '':
        row["pickup_hour"] = parser.parse(row["pickup_datetime"]).hour
    else:
        row['pickup_hour']=mean_pickuphour
    if row["pickup_datetime"]!="" and row["dropoff_datetime"]!="":
        row["total_time"]=(parser.parse(row["dropoff_datetime"]) - parser.parse(row["pickup_datetime"])).seconds/60
    else:
        row["total_time"]=mean_totaltime
    return row

#dataset=dataset.apply(lambda row: copy(row),axis=1)
#dataset.to_csv("Data/clustered_train.csv",sep=",")
#testset.loc[testset.surcharge.isnull(),'surcharge']=0
#testset.loc[testset.tip_amount.isnull(),'tip_amount']=0
#testset['payment_type']=testset['payment_type'].map({'NOC ':0,'CSH': 1, 'CRD': 2,'DIS ':3,'UNK ':4})
#testset.to_csv("Data/cleanedData.csv",sep=",")

#t_train = dataset.sample(frac=0.1,replace=False)



#kmeans = KMeans(n_clusters=100, random_state=0).fit(dataset[["pickup_latitude","pickup_longitude"]].sample(frac=0.3))



#t_train = dataset.sample(frac=0.1,replace=False)

#imp = Imputer(missing_values='NaN', strategy='mean', axis=0)


'''
t_X = t_train[["pickup_longitude","dropoff_latitude","dropoff_longitude"]]
t_Y = t_train["pickup_latitude"]
t_rtr = RandomForestRegressor( n_jobs=-1).fit(t_X, t_Y)
t_test = testset.loc[(testset.pickup_latitude.isnull())]
imp=imp.fit(t_test[["pickup_longitude","dropoff_latitude","dropoff_longitude"]])
t_pred = t_rtr.predict(imp.transform(t_test[["pickup_longitude","dropoff_latitude","dropoff_longitude"]]))
testset.loc[(testset.pickup_latitude.isnull()), 'pickup_latitude'] =t_pred
testset.to_csv("Data/cleanedData.csv",sep=",")

t_X = t_train[["pickup_latitude","dropoff_latitude","dropoff_longitude"]]
t_Y = t_train["pickup_longitude"]
t_rtr = RandomForestRegressor( n_jobs=-1).fit(t_X, t_Y)
t_test = testset.loc[(testset.pickup_longitude.isnull())]
imp=imp.fit(t_test[["pickup_latitude","dropoff_latitude","dropoff_longitude"]])
t_pred = t_rtr.predict(imp.transform(t_test[["pickup_latitude","dropoff_latitude","dropoff_longitude"]]))
testset.loc[(testset.pickup_longitude.isnull()), 'pickup_longitude'] =t_pred
testset.to_csv("Data/cleanedData.csv",sep=",")


t_test = testset.loc[(testset.payment_type.isnull())]
t_train = dataset.sample(frac=0.1,replace=False)
t_X = t_train[["tip_amount","surcharge","mta_tax"]]
t_Y = t_train["payment_type"]
t_rtr = RandomForestRegressor( n_jobs=-1).fit(t_X, t_Y)
t_pred = t_rtr.predict(t_test[["tip_amount","surcharge","mta_tax"]])
testset.loc[(testset.payment_type.isnull()), 'payment_type'] =t_pred
testset.to_csv("Data/cleanedData.csv",sep=",")
'''

#testset=testset.apply(lambda row: copy(row),axis=1)
#testset.to_csv("Data/cleanedData.csv",sep=",")

#testset["pickup_location"]=kmeans.predict(testset[["pickup_latitude","pickup_longitude"]])


for i in range(0,100):
    mean=np.mean(dataset[dataset.pickup_location==i]["avg_fare"])
    if np.isnan(mean):
        mean=np.median(dataset["avg_fare"])
    testset.loc[(testset.pickup_location==i),"avg_fare"]=mean

testset.to_csv("Data/cleanedData.csv",sep=",")
print(testset.describe().T)
print(testset.corr())

regr = linear_model.SGDRegressor(alpha=0.00001,n_iter=1000)
X = dataset[listOfColumnsInConsideration]
Y = dataset["fare_amount"]
regr.fit(X, Y)

X=testset[listOfColumnsInConsideration]
predictions=regr.predict(X)
testset["fare_amount"]=predictions
testset[["TID","fare_amount"]].to_csv("outcome_random.csv",sep=",")