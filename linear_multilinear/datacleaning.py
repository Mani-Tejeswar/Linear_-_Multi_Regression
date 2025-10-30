import pandas as pd 

dataset=pd.read_csv("realistic_body_data.csv")

#print(dataset.head())

#print(dataset.tail())
#print(dataset.info())
#print(dataset.describe())
#print(dataset.columns)
#print(dataset.dtypes)
#print(dataset.shape)

print(dataset.isnull().sum())


dataset[["height","weight","waist","chest","hips"]].isnull().sum()
x,y,z,a,b=[dataset[col].mean() for col in ["height","weight","waist","hips","chest"]]
print(x,y,z,a,b)
dataset.fillna({"height":x,"weight":y,"waist":z,"hips":a,"chest":b},inplace=True)
print(dataset.isnull().sum())

    








