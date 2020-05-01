# AUTHOR - IESHAAN SHARMAA


# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

# READ DATASET

hdata = pd.read_csv(r"C:\Users\A K SHARMA\Downloads\hotel_bookings.csv",encoding='utf8')

hdata.head()

# REPLACE MISSING VALUES BY INTERPOLATION

hdata.interpolate(method ='linear', limit_direction ='forward') 

# CHECK NO OF NULL VALUES IN EACH COLUMN

hdata.isnull().sum()

# FILL NULL VALUES BY MEAN AND MEDIAN 

hdata['agent']=hdata['agent'].fillna(hdata['agent'].mean())

hdata['company']=hdata['company'].fillna(hdata['company'].median())

hdata.isnull().sum()

# SPLIT THE DEPENDENT AND INDEPENDENT VARIABLES

X = hdata.drop(columns=['is_canceled'],axis=1)

Y = hdata['is_canceled']

hdata.head(5)

# LABEL ENCODING TO CONVERT CATEOGORICAL VARIABLES TO NUMERIC 

from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
X['hotel']= label_encoder.fit_transform(X['hotel']) 

X['arrival_date_month']= label_encoder.fit_transform(X['arrival_date_month'])
X['deposit_type']= label_encoder.fit_transform(X['deposit_type'])
X['assigned_room_type']= label_encoder.fit_transform(X['assigned_room_type'])
X['deposit_type']= label_encoder.fit_transform(X['deposit_type'])
X['customer_type']= label_encoder.fit_transform(X['customer_type'])
X['reservation_status']= label_encoder.fit_transform(X['reservation_status'])

X['market_segment']= label_encoder.fit_transform(X['market_segment'])
X['distribution_channel']= label_encoder.fit_transform(X['distribution_channel'])
X['reserved_room_type']= label_encoder.fit_transform(X['reserved_room_type'])

# YOU CAN CHECK UNIQUE VALUES IN EACH COLUMN TO CHECK NAN VALUES

X['meal'].unique()
X['country'].unique()

X['country'] = X['country'].astype(str)
X['country'] = label_encoder.fit_transform(X['country'])

X['children']=X['children'].fillna(X['children'].mean())
X.isna().sum()

#get correlations of each features in dataset

corrmat = hdata.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(hdata[top_corr_features].corr(),annot=True,cmap="RdYlGn")

X.shape

E = X[['lead_time','arrival_date_year','arrival_date_week_number','stays_in_weekend_nights','adults','children','previous_cancellations','adr','days_in_waiting_list']]
F = hdata['is_canceled']

# SPLIT DATA INTO TRAINING AND TESTING DATASET 

from sklearn.model_selection import train_test_split

E_train, E_test, F_train, F_test = train_test_split(E, F, test_size = 0.2, random_state = 10)


# RANDOM FOREST

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(E_train, F_train)

F_pred_RF = random_forest.predict(E_test)

random_forest.score(E_train, F_train)
acc_random_forest = round(random_forest.score(E_train, F_train) * 100, 2)

# LOGISTIC REGRESSION

logreg = LogisticRegression()
logreg.fit(E_train, F_train)

F_pred_LR = logreg.predict(E_test)

acc_log = round(logreg.score(E_train, F_train) * 100, 2)

# K NEAREST NEIGHBOURS

knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(E_train, F_train) 
F_pred_KNN = knn.predict(E_test)  
acc_knn = round(knn.score(E_train, F_train) * 100, 2)

# GAUSSIAN NAIVE BAYES

gaussian = GaussianNB()
gaussian.fit(E_train, F_train) 
F_pred_NB = gaussian.predict(E_test)  
acc_gaussian = round(gaussian.score(E_train, F_train) * 100, 2)

#Linear Support Vector Machine:

linear_svc = LinearSVC()
linear_svc.fit(E_train, F_train)

F_pred_SVM = linear_svc.predict(E_test)

acc_linear_svc = round(linear_svc.score(E_train, F_train) * 100, 2)

# Decision Tree
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(E_train, F_train)  
F_pred_DT = decision_tree.predict(E_test)  
acc_decision_tree = round(decision_tree.score(E_train, F_train) * 100, 2)

# PRINT THE ACCURACY OF EACH ALGORITHM IN  A TABULAR FORMAT

results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes','Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian,acc_decision_tree]}
)
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)
