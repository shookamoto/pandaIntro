#Pandas intro




import pandas as pd
import numpy as np
import pylab as P

from sklearn.ensemble import RandomForestClassifier as rfc

#garbage collection
import gc as gc
#collects any unreferenced objects
gc.collect()

#clear the variable
t=1
t= None

#delete the variable
t= 1
del t



dataPath = '/Users/sho/Documents/python/pandas/pandasIntro/'

df=pd.read_csv(dataPath +'train.csv')

#head
df.head(10)
#check var types
df.dtypes

#check for missing obs
df.info()

#summary stats for numeric vars
df.describe()

#data access
df['Age']
df.Age

df['Age'].head(5)

df['Age'][0:5]


#type checking
type(df['Age'])

#get the mean/median/etc
df['Age'].mean()
df['Age'].median()
df['Age'].min()
df['Age'].max()
df['Age'].quantile(0.5)

#get headers
df.columns.values


#subsetting
df[['Sex', 'Pclass','Age' ]]

#filtering (usual stuff)
df[df['Age']>50]

#missing values
#returns a boolean array
df['Age'].isnull()

#subsetting for missing values
df[df['Age'].notnull()][['Sex', 'Pclass', 'Age']]
df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]

#count
len(df[df['Age'].isnull()])
len(df[df['Age'].notnull()])


#hist,
df['Age'].hist()

#drop na
df['Age'].dropna()
len(df['Age'].dropna())

df['Age'].hist(bins=10, range=(0,80), alpha=0.5)

P.show()

#adding variables
df['Gender']=3

#map function and anonymous functions
#lambda allows us to write anonymous functions on the fly
#map applies a function to each of the correspoinding values, basically like apply in R

#list (ordered, equivalent)
r = [1,2,3]
#tupple (unordered, heterogenous, sort of.... not really sure of distinction..)
#tuples are immutable, lists are mutable.
r=(1,'a',3)

r=[1,2,3]

r=(1,2,3)
map(lambda x: x*2-1, r)

#variable replacement
#uppers the first character of the gender
df['Gender'] = df['Sex'].map(lambda x: x[0].upper())

#string is an array?
t="this is a test"
t[3]

for i in range(len(t)):
    print t[i]

df['Gender']= df['Sex'].map({'female': 0, 'male': 1}).astype(int)

#fill in ages with medians for each class values and gender
median_ages = np.zeros((2,3))

for i in range(2):
    for j in range(3):
        median_ages[i,j] = df[(df['Gender']==i) & (df['Pclass']==j+1)]['Age'].dropna().median()


#make a copy of age
df['AgeFill']=df['Age']

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]



df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)

#record whether the age was missing in the first place
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

#Parch is the number of parents on board
#Sibsp is the number of siblings or spouses

df['FamilySize']= df['SibSp']+df.Parch

#add an interaction term
df['Age*Class'] = df.AgeFill * df.Pclass

df.dtypes

#apply to columns that are not numeric.
df.dtypes[df.dtypes.map(lambda x: x=='object' )]

#drop columns that are not used
#assume axis is like dim in R
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'PassengerId'], axis=1)

#similar to complete cases in R
df.dropna()


#general point, these functions appear to work on value copies rather than to references...

#store out a vector of names
header = df.columns.values  



#turn back into a numpy array, for sklearn?
train_data = df.values


#sklearn
#methods share common syntax (good!)

#some-model-name.fit( )

#some-model-name.predict( )

#some-model-name.score( )


#Random Forrest
#create the random forrest object which will include all the parameters

forest = rfc(n_estimators=100)

#Fit the training data to the training set, use the Survivied as the Predictor
forest_fit = forest.fit(train_data[0::, 1::], train_data[0::, 0])
