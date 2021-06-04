import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
subject_id =  dataset["SubjectID"].copy() #back up SubjectID column
dataset = dataset.drop("SubjectID",axis=1) #drop SubjectID column
dataset = clean_data(dataset) #Encode dataset column
dataset = dataset[dataset.applymap(isnumber)] 
dataset = dataset.apply(lambda x: x.fillna(x.mean()), axis=0) #fill the missing values with mean value of that column

df = pd.DataFrame(dataset)

# Plot Before

df['Metabolic syndrome'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Metabolic syndrome", labelpad=14)
plt.ylabel("Count of People", labelpad=14)
plt.title("Count of People that have matabolic syndrome", y=1.02);
#plt.show()

#Oversampling the data
smote = SMOTE(random_state = 101)
X, y = smote.fit_resample(df[df.columns.difference(['Metabolic syndrome'])], df['Metabolic syndrome'])

#Creating a new Oversampling Data Frame
df_oversampler = pd.DataFrame(X, columns = ['Gender','Age','Height','Weight','BMI','High blood pressure','bodyfat','Triglyceride','HDL cholesterol','Blood sugar','Drinking','walking_min','walking_num_per_day'])
df_oversampler['Metabolic syndrome'] = y
#print(df_oversampler)
df_oversampler['Metabolic syndrome'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Metabolic syndrome", labelpad=14)
plt.ylabel("Count of People", labelpad=14)
plt.title("Count of People that have matabolic syndrome", y=1.02);
#plt.show()

training_set, test_set = train_test_split(df_oversampler, test_size = 0.3) #split train and test dataset

train_features, train_labels = split_feature_class(training_set, 'Metabolic syndrome') #split train features and labels
test_features, test_labels = split_feature_class(test_set, 'Metabolic syndrome') #split test features and labels


#print(dataset)
#print(test_features)

#create model and fit it with train dataset
model = RandomForestClassifier()
model.fit(train_features, train_labels)

#test the model by predicting the test dataset and show the probability
predict = model.predict(test_features)
predict_prob = model.predict_proba(test_features)
print("Prediction of test_set: ",predict)
print("Probabilitis: \n",predict_prob)

print("Accuracy = ", accuracy_score(test_labels, predict)) #Calculate the accuracy
print(classification_report(test_labels, model.predict(test_features)))

cm = confusion_matrix(test_labels, predict, normalize='all')
cmd = ConfusionMatrixDisplay(cm, display_labels=['1','0'])
cmd.plot()

cm_df = pd.DataFrame(cm,
                 index = ['1', '0'], 
                 columns = ['1', '0'])
ax = sns.heatmap(cm_df, cmap="YlGnBu", cbar = False,  annot=True)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
#plt.show()

#Test with new data
dataset_test = pd.read_csv("dataset_test.csv") #data filled with estimated values
subject_id =  dataset_test["SubjectID"].copy() #back up SubjectID column
dataset_test = dataset_test.drop("SubjectID",axis=1) #drop SubjectID column
dataset_test = clean_data(dataset_test) #Encode dataset column
dataset_test = dataset_test[dataset_test.applymap(isnumber)] 
dataset_test = dataset_test.apply(lambda x: x.fillna(x.mean()), axis=0) #fill the missing values with mean value of that column

df_test = pd.DataFrame(dataset_test)

predict_prob = model.predict_proba(df_test)
print("Probabilitis: \n",predict_prob)
print("Probability that ",subject_id[0]," has Metabolic syndrome is ",predict_prob[0][1] * 100, "%")