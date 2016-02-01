import csv as csv
import numpy as np



dataPath = '/Users/sho/Documents/python/pandas/pandasIntro/'

csv_file_object = csv.reader(open(dataPath +'train.csv'))

header1 = csv_file_object.next()
data=[]

for row in csv_file_object:
    data.append(row)

data=np.array(data)

#get the number of passengers
number_passengers = np.size(data[0::, 0].astype(float))
number_survived = np.sum(data[0::, 1].astype(int))
proportion_survivors = number_survived.astype(float)/number_passengers
women_only_stats = data[0::,4]=='female'
men_only_stats = data[0::, 4]!='female'

women_onboard = data[women_only_stats, 1].astype(float)
men_onboard = data[men_only_stats, 1].astype(float)
prop_women_survived = np.sum(women_onboard)/np.size(women_onboard)
prop_men_survived = np.sum(men_onboard)/np.size(men_onboard)

test_file = open(dataPath + 'test.csv')
test_file_object = csv.reader(test_file)
header2 = test_file_object.next()

#open a pointer to a new file so that we can write to it
prediction_file = open(dataPath+"genderBaseModel.csv", 'wb')
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerID", "Survived"])

for row in test_file_object:
    survived = 0
    if row[3]=='female': survived = 1

    prediction_file_object.writerow([row[0], survived])

test_file.close()
prediction_file.close()

#part 2
fare_ceiling = 40
#modify so that any fares above 39 is set to 39
data[(data[0::, 9].astype(float)>39),9]= fare_ceiling-1

fare_bracket_size = 10
number_of_price_buckets = fare_ceiling/fare_bracket_size

#number of classes
number_of_classes = len(np.unique(data[0::, 2]))

#initialise the survival table with all 0s
survival_table = np.zeros((2, number_of_classes, number_of_price_buckets))

women_only_stats=[]
men_only_stats=[]


for i in xrange(number_of_classes):
    for j in xrange(number_of_price_buckets):
        #should really be looped....
        women_only_stats = data[(data[0::, 4] =='female') & \
                           (data[0::,2].astype(np.float)==i+1) & \
                           (data[0::, 9].astype(np.float)< (j+1)*fare_bracket_size) & \
                           (data[0::, 9].astype(np.float)>= (j)* fare_bracket_size),1]

        men_only_stats = data[(data[0::, 4] !='female') & \
                         (data[0::,2].astype(np.float)==i+1) & \
                         (data[0::, 9].astype(np.float)< (j+1)*fare_bracket_size) & \
                         (data[0::, 9].astype(np.float)>= (j)* fare_bracket_size),1]

        survival_table[ survival_table != survival_table ] = 0.

        if np.size(women_only_stats)==0:
            survival_table[0,i,j]= 0
        else:
             survival_table[0,i,j] = np.mean(women_only_stats.astype(float))

        if np.size(men_only_stats)==0:
            survival_table[1,i,j]= 0
        else:
             survival_table[1,i,j] = np.mean(men_only_stats.astype(float))


survival_table[survival_table>0.5]=1
survival_table[survival_table<=0.5]=0


test_file = open(dataPath + 'test.csv')
test_file_object = csv.reader(test_file)
header2 = test_file_object.next()

#open a pointer to a new file so that we can write to it
prediction_file = open(dataPath+"genderClassModel.csv", 'wb')
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerID", "Survived"])


for row in test_file_object:


    #assign a bin to each customer
    bin_fare=0
    for j in xrange(number_of_price_buckets):

        try:
            row[8]=float(row[8])
        except:
            #if no price data
            print(row[1])
            bin_fare= 3-float(row[1])
            break

        if (row[8]>=fare_ceiling):
            bin_fare=number_of_price_buckets-1
            break

        if row[8] >= j * fare_bracket_size and row[8] < (j+1) * fare_bracket_size:
            bin_fare = j
            break

    if (row[3]=='female'):
        gender = 0
    else:
        gender = 1

    prediction_file_object.writerow([row[0], "%d"% int(survival_table[gender,float(row[1])-1, bin_fare])])

test_file.close()




