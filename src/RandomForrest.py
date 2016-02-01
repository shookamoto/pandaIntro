#quick aggregation of stuff learned.
#first clean the data in pandas and then run the random forest algorithm on the data;

import pandas as pd
import numpy as np
import pylab as P
from sklearn.ensemble import RandomForestClassifier as rfc
import pickle



#create a routine that returns clean data set,
#the only difference should be the test or train set, but not a good idea to automate cleaning of traiin and test
#instead write functions to automate aspects of the cleaning


#constants
dataPath = '/Users/sho/Documents/python/pandas/pandasIntro/'


#are objects passed as reference or as values?
#train
df_train= None
df_train = pd.read_csv(dataPath + 'train.csv')




def mutationTest(df):
    df['Added']=1


mutationTest(df_train)

#has been mutated, objects are sent via pointers
assert 'Added' in df_train.columns.values

df_train=None



#simple function that automates getting the median age for each class and gender

def fillMissingAges(df):
    #initialise the return array
    median_ages= np.zeros((2,3))

    for i in range(2):
        for j in range(3):
            median_ages[i,j] = df[(df['Gender']==i) & (df['Pclass']==j+1)]['Age'].dropna().median()


    #make a copy of age
    df['AgeFill']=df['Age']

    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'AgeFill'] = median_ages[i,j]

    return None



i


df=[]
#ordering is training and then test data set
for i in range(2):
    if (not i):
        df.append( pd.read_csv(dataPath + 'train.csv'))
    else:
        df.append(pd.read_csv(dataPath + 'test.csv'))


#objects are stored as references,
#equality sets a pointer, not a copy
#test
data = df[1]
data["TEST"]=3
assert "TEST" in df[1].columns.values
data.drop(['TEST'],axis=1)




#clean the data
finalData = []

for i in range(2):
    data  = df[i]

    #encode the gender data
    data['Gender']= data['Sex'].map({'female': 0, 'male': 1}).astype(int)

    #encode the Embarked data
    data['Embarked'] = data['Embarked'].dropna().map({'S':0, 'C':1, 'Q':2}).astype(int)

    fillMissingAges(data)

    #record whether the age was missing in the first place
    data['AgeIsNull'] = pd.isnull(data.Age).astype(int)

    #Parch is the number of parents on board
    #Sibsp is the number of siblings or spouses

    data['FamilySize']= data['SibSp']+data.Parch

    #add an interaction term
    data['Age_Class'] = data.AgeFill * data.Pclass

    if (not i):
        keepVars = ['Survived', 'Pclass', 'Parch', 'Fare', 'Gender', 'AgeFill','AgeIsNull', 'FamilySize', 'Age_Class']

        #save the headers as well...
        headers = keepVars

    else:
        #test set doesn't contain the outcome variable
        keepVars=['Pclass', 'Parch', 'Fare', 'Gender', 'AgeFill','AgeIsNull', 'FamilySize', 'Age_Class']
    data=data[keepVars]

    #drop missing values
    data = data.dropna()

    finalData.append(data.values)


#save the data
saveFile = open(dataPath+'inputData', 'wb')
pickle.dump(finalData, saveFile)
saveFile.close()



finalData=None
loadFile = open(dataPath+'inputData', 'rb')
finalData=pickle.load(loadFile)

#now separate out into test and training data
data_train = finalData[0]
data_test = finalData[1]


#sklearn
#methods share common syntax (good!)

#some-model-name.fit( )

#some-model-name.predict( )

#some-model-name.score( )


#Random Forrest
#create the random forrest object which will include all the parameters

forest = rfc(n_estimators=100)


#Fit the training data to the training set, use the Survivied as the Predictor
forest_fit = forest.fit(data_train[0::, 1::], data_train[0::, 0])

#make a prediction
output = forest.predict(data_test)
