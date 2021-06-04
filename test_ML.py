import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def clean_data(dataset): #Encode every column that isn't numeric
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
    return dataset

def split_feature_class(dataset, feature): #Split features and labels to 2 datasets
    features = dataset.drop(feature,axis=1)
    labels =  dataset[feature].copy()
    return features, labels

def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False

#dataset = pd.read_csv("dataset_raw.csv") #data with missing values
dataset = pd.read_csv("dataset_plus.csv") #data filled with estimated values

'''test = pd.read_csv("dataset_test.csv") #these line are for specific test data
test = test.drop("SubjectID",axis=1) #drop SubjectID column
test = test.drop('Metabolic syndrome',axis=1)
test = clean_data(test)
#print(test)'''

subject_id =  dataset["SubjectID"].copy() #back up SubjectID column
dataset = dataset.drop("SubjectID",axis=1) #drop SubjectID column
dataset = clean_data(dataset) #Encode dataset column
dataset = dataset[dataset.applymap(isnumber)] 
dataset = dataset.apply(lambda x: x.fillna(x.mean()), axis=0) #fill the missing values with mean value of that column


training_set, test_set = train_test_split(dataset)#, test_size = 0.25) #split train and test dataset

train_features, train_labels = split_feature_class(training_set, 'Metabolic syndrome') #split train features and labels
test_features, test_labels = split_feature_class(test_set, 'Metabolic syndrome') #split test features and labels


print(dataset)
print(test_features)

#create model and fit it with train dataset
model = RandomForestClassifier()
model.fit(train_features, train_labels)

#test the model by predicting the test dataset and show the probability
predict = model.predict(test_features)
predict_prob = model.predict_proba(test_features)
print("Prediction of test_set: ",predict)
print("Probabilitis: \n",predict_prob)

print("Accuracy = ", accuracy_score(test_labels, predict)) #Calculate the accuracy