import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, auc,roc_curve

Churn_Data=pd.read_csv(r"C:\Users\91831\Desktop\Auth_Copy\website\Customer Churn.csv")

Churn_Data.head(10)

Churn_Data.shape

Churn_Data.columns

Churn_Data.isna().sum()

Churn_Data.dtypes

Churn_Data.describe()
Churn_Data.loc[Churn_Data.duplicated()]

Churn_Data.drop_duplicates().shape
Churn_Data.nunique()

df = pd.read_csv(r"C:\Users\91831\Desktop\Auth_Copy\website\Customer Churn.csv",sep='\s+')

pat = '[(:/,#%\=@)]'
df['count'] = df['Call'].str.count(pat)
df

start = 1
df.insert(0, 'ID', range(start, start + df.shape[0]))
df[df['count'] > 13].groupby('ID')['Call'].count()

from statsmodels.stats.outliers_influence import variance_inflation_factor
X = Churn_Data[['Call  Failure', 'Complains', 'Subscription  Length', 'Charge  Amount',
       'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
       'Distinct Called Numbers', 'Age Group', 'Tariff Plan', 'Status', 'Age',
       'Customer Value', 'Churn']]

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)

sns.pairplot(Churn_Data)

f,ax=plt.subplots(1,1,figsize=(19,10))
sns.heatmap(Churn_Data.corr(),annot=True)
ax.set_title('Correlation coefficiency of All features ',fontsize=16)

def correl(X_train):
    cor = X_train.corr()
    corrm = np.corrcoef(X_train.transpose())
    corr = corrm - np.diagflat(corrm.diagonal())
    print("max corr:",corr.max(), ", min corr: ", corr.min())
    c1 = cor.stack().sort_values(ascending=False).drop_duplicates()
    high_cor = c1[c1.values!=1]        
    thresh = 0.9
    display(high_cor[high_cor>thresh])
correl(Churn_Data)

Churn_Data.shape

corr_matrix = Churn_Data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
to_drop

Churn_Data.drop(to_drop, axis=1, inplace=True)
Churn_Data

Q1 = Churn_Data.quantile(0.25)
Q3 = Churn_Data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

print(Churn_Data['Seconds of Use'].skew())
Churn_Data['Seconds of Use'].describe()

sns.boxplot(data=Churn_Data[["Seconds of Use"]], orient="h")
print(Churn_Data['Frequency of SMS'].skew())
Churn_Data['Frequency of SMS'].describe()

sns.boxplot(data=Churn_Data[["Frequency of SMS"]], orient="h")

fig, ax = plt.subplots(figsize=(12,6))
ax.scatter(Churn_Data['Seconds of Use'], Churn_Data['Frequency of SMS'])
ax.set_xlabel('Seconds of Use ')
ax.set_ylabel('Frequency of SMS')
plt.show()

print(Churn_Data['Distinct Called Numbers'].skew())
Churn_Data['Distinct Called Numbers'].describe()

Box_plt=Churn_Data.drop(['Seconds of Use','Frequency of SMS'],axis=1)
boxplot = Box_plt.boxplot(grid=False, rot=45, fontsize=10,figsize=(10,10))  
#sns.set(rc={"figure.figsize":(20, 20)})
plt.figure(figsize=(13,10))
sns.boxplot(data=Box_plt, orient="h",width=0.70)
Churn_Data_=Churn_Data

print(Churn_Data_['Seconds of Use'].quantile(0.10))
print(Churn_Data_['Seconds of Use'].quantile(0.90))

Churn_Data_['Seconds of Use'] = np.where(Churn_Data_['Seconds of Use'] <510.0, 510.0,Churn_Data_['Seconds of Use'])
Churn_Data_["Seconds of Use"] = np.where(Churn_Data_["Seconds of Use"] >10933.0, 10933.0,Churn_Data_['Seconds of Use'])
print(Churn_Data_['Seconds of Use'].skew())


print(Churn_Data_['Frequency of SMS'].quantile(0.10))
print(Churn_Data_['Frequency of SMS'].quantile(0.90))

Churn_Data_['Frequency of SMS'] = np.where(Churn_Data_['Frequency of SMS'] <0.0, 0.0,Churn_Data_['Frequency of SMS'])
Churn_Data_["Frequency of SMS"] = np.where(Churn_Data_["Frequency of SMS"] >240.0, 240.0,Churn_Data_['Frequency of SMS'])
print(Churn_Data_['Frequency of SMS'].skew())

Churn_Data_.shape

df_out = Churn_Data_[~((Churn_Data_ < (Q1 - 1.5 * IQR)) |(Churn_Data_ > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df_out.shape)

df_out.describe()

Churn_Data_["Log_Freq"] = Churn_Data_["Frequency of SMS"].map(lambda i: np.log(i) if i > 0 else 0) 
print(Churn_Data_['Frequency of SMS'].skew())
print(Churn_Data_['Log_Freq'].skew())

Churn_Data_["Log_use"] = Churn_Data_["Seconds of Use"].map(lambda i: np.log(i) if i > 0 else 0) 
print(Churn_Data_['Seconds of Use'].skew())
print(Churn_Data_['Log_use'].skew())

print(Churn_Data_['Frequency of SMS'].quantile(0.50)) 
print(Churn_Data_['Frequency of SMS'].quantile(0.95)) 
Churn_Data_['Frequency of SMS'] = np.where(Churn_Data_['Frequency of SMS'] > 325, 140, Churn_Data_['Frequency of SMS'])
Churn_Data_.describe()

Churn_Data_=Churn_Data_.drop(['Seconds of Use','Frequency of SMS'],axis=1)


Churn=Churn_Data_['Churn']
Churn=pd.get_dummies(Churn)
Churn.rename(columns ={0: 'Not_Churn',1: 'Churn'}, inplace = True)
Churn=Churn.sum()
Churn=pd.DataFrame(Churn)

from sklearn.utils import resample
Num_Y=Churn_Data_['Churn']
Num_X=Churn_Data_.drop(['Churn'],axis=1)
a_train, a_test, b_train, b_test = train_test_split(Num_X, Num_Y, test_size=0.25, random_state=27)


a = pd.concat([a_train, b_train], axis=1)
not_Churn = a[a['Churn']==0]
Churn = a[a['Churn']==1]


# In[46]:


Churn_upsampled = resample(Churn,
                          replace=True,
                          n_samples=len(not_Churn),
                          random_state=27) 


# In[47]:


upsampled = pd.concat([not_Churn, Churn_upsampled])
upsampled.Churn.value_counts()


# In[48]:


Churn=upsampled['Churn']
Churn=pd.get_dummies(Churn)
Churn.rename(columns ={0: 'Not_Churn',1: 'Churn'}, inplace = True)
Churn=Churn.sum()
Churn=pd.DataFrame(Churn)


# # Standardizing 

# In[49]:


Y_train = upsampled.Churn
X_train = upsampled.drop('Churn', axis=1)


# In[50]:


Y_train=pd.DataFrame(Y_train)


# In[51]:


start = 1
Y_train.insert(0, 'ID', range(start, start + Y_train.shape[0]))


# In[52]:


from sklearn.preprocessing import StandardScaler
object= StandardScaler()
scale = object.fit_transform(X_train) 
print(scale)


# In[53]:


df=pd.DataFrame(scale)


# In[54]:


start = 1
df.insert(0, 'ID', range(start, start + df.shape[0]))


# In[55]:


df = pd.merge(df, Y_train, on='ID')


# In[56]:


df=df.drop('ID', axis=1)


# In[57]:


Y_train = df.Churn
X_train = df.drop('Churn', axis=1)


# In[58]:


X=X_train
Y=Y_train


# In[59]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=30)


# # PCA

# In[60]:


scaler=StandardScaler()
scaler.fit(upsampled)
X_scaler=scaler.transform(upsampled)


# In[61]:


from sklearn.decomposition import PCA
pca=PCA(n_components=11,random_state=20)
pca.fit(X_scaler)
pca_=pca.transform(X_scaler)
pca.explained_variance_ratio_*100


# In[62]:


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Explined variance')


# # Logestic Regression

# In[63]:


logreg = LogisticRegression(random_state=16)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


# ## Hyper Parameter Tuning 

# In[64]:


clf = [
    LogisticRegression(solver='newton-cg',penalty='l2',max_iter=1000),
    LogisticRegression(solver='lbfgs',penalty='l2',max_iter=1000),
    LogisticRegression(solver='sag',penalty='l2',max_iter=1000),
    LogisticRegression(solver='saga',penalty='l2',max_iter=1000)
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# In[65]:


clf = [
    LogisticRegression(solver='newton-cg',penalty='l2', C=0.001, max_iter=1000),
    LogisticRegression(solver='lbfgs',penalty='l2',C=0.001, max_iter=1000),
    LogisticRegression(solver='sag',penalty='l2',C=0.001, max_iter=1000),
    LogisticRegression(solver='saga',penalty='l2',C=0.001, max_iter=1000)
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    clf_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted),5)
    clf_compare.loc[row_index, 'AUC'] = round(auc(fp, tp),5)

    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare


# In[66]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[67]:


#Plot the confusion matrix.
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['Churn','Not Churn'],
            yticklabels=['Churn','Not Churn'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()


# In[68]:


class_report = classification_report(y_test,y_pred)
print(class_report)


# # KNN

# In[69]:


knn = KNeighborsClassifier()


# In[70]:


# calculating the accuracy of models with different values of k
mean_acc = np.zeros(20)
for i in range(1,21):
    #Train Model and Predict  
    knn = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat= knn.predict(X_test)
    mean_acc[i-1] = metrics.accuracy_score(y_test, yhat)


# In[71]:


mean_acc


# In[72]:


loc = np.arange(1,21,step=1.0)
plt.figure(figsize = (10, 6))
plt.plot(range(1,21), mean_acc)
plt.xticks(loc)
plt.xlabel('Number of Neighbors ')
plt.ylabel('Accuracy')
plt.show()


# # Hyper Parameter Tuning For KNN

# In[73]:


from sklearn.model_selection import GridSearchCV
grid_params = { 'n_neighbors' : [5,7,9,11,13,15,1,17,20],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1)


# In[74]:


# fit the model on our train set
g_res = gs.fit(X_train, y_train)


# In[75]:


# find the best score
g_res.best_score_


# In[76]:


# get the hyperparameters with the best score
g_res.best_params_


# In[77]:


# use the best hyperparameters
knn = KNeighborsClassifier(n_neighbors = 1, weights = 'uniform',metric = 'manhattan')
knn.fit(X_train, y_train)


# In[78]:


# get a prediction
y_hat = knn.predict(X_train)
y_knn = knn.predict(X_test)


# In[79]:


print('Training set accuracy: ', metrics.accuracy_score(y_train, y_hat))
print('Test set accuracy: ',metrics.accuracy_score(y_test, y_knn))


# In[80]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_knn))


# In[81]:


from sklearn.metrics import confusion_matrix
#compute the confusion matrix.
cm = confusion_matrix(y_test,y_knn)
 
#Plot the confusion matrix.
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['Churn','Not Churn'],
            yticklabels=['Churn','Not Churn'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=15)
plt.show()


# # Naive Bayes

# In[82]:


# Gaussian Naive Bayes Classification
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings("ignore")
import scipy.stats as stats
model = GaussianNB()
cv_scores = cross_val_score(model, X, Y, cv=5)
    
print(model, ' mean accuracy: ', round(cv_scores.mean()*100, 3), '% std: ', round(cv_scores.var()*100, 3),'%')


# In[83]:


predict_train = model.fit(X_train, y_train).predict(X_train)
from sklearn.metrics import accuracy_score, confusion_matrix
# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)


# predict the target on the test dataset
predict_test = model.predict(X_test)

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_test,predict_test)
print('accuracy_score on test dataset : ', accuracy_test)


# In[84]:


from sklearn import metrics
f,a =  plt.subplots(1,2,sharex=True,sharey=True,squeeze=False)

#Plotting confusion matrix for the different models for the Training Data

plot_0 = sns.heatmap((metrics.confusion_matrix(y_train,predict_train)),annot=True,fmt='.5g',cmap="YlGn",ax=a[0][0]);
a[0][0].set_title('Training Data')

plot_1 = sns.heatmap((metrics.confusion_matrix(y_test,predict_test)),annot=True,fmt='.5g',cmap="YlGn",ax=a[0][1]);
a[0][1].set_title('Test Data');


# In[85]:


from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix
print(classification_report(y_train,predict_train))
print(classification_report(y_test,predict_test))


# ## Hyper Parameter Tuning

# In[86]:


np.logspace(0,-9, num=10)
from sklearn.model_selection import RepeatedStratifiedKFold

cv_method = RepeatedStratifiedKFold(n_splits=5, 
                                    n_repeats=3, 
                                    random_state=999)


# In[87]:


from sklearn.preprocessing import PowerTransformer
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}

gs_NB = GridSearchCV(estimator=model, 
                     param_grid=params_NB, 
                     cv=cv_method,
                     verbose=1, 
                     scoring='accuracy')

Data_transformed = PowerTransformer().fit_transform(X_test)

gs_NB.fit(Data_transformed, y_test);


# In[88]:


gs_NB.best_params_


# In[89]:


gs_NB.best_score_


# In[90]:


results_NB = pd.DataFrame(gs_NB.cv_results_['params'])
results_NB['test_score'] = gs_NB.cv_results_['mean_test_score']
plt.plot(results_NB['var_smoothing'], results_NB['test_score'], marker = '.')    
plt.xlabel('Var. Smoothing')
plt.ylabel("Mean CV Score")
plt.title("NB Performance Comparison")
plt.show()


# In[91]:


# predict the target on the test dataset
predict_test = gs_NB.predict(Data_transformed)

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_test,predict_test)
print('accuracy_score on test dataset : ', accuracy_test)


# In[92]:


sns.heatmap((metrics.confusion_matrix(y_test,predict_test)),annot=True,fmt='.5g',cmap="YlGn").set_title('Test Data');

from sklearn.svm import SVC 

model = SVC() 
model.fit(X_train, y_train) 

predictions = model.predict(X_test) 
print(classification_report(y_test, predictions)) 

from sklearn.model_selection import GridSearchCV 

param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
 
grid.fit(X_train, y_train) 

print(grid.best_params_)   

print(grid.best_estimator_) 

grid_predictions = grid.predict(X_test)   

print(classification_report(y_test, grid_predictions)) 

cm = confusion_matrix(y_test,grid_predictions)
 
#Plot the confusion matrix.
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['Fraud','Not Fraud'],
            yticklabels=['Fraud','Not Fraud'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=15)
plt.show()


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

param_dist = {"max_depth": [3, 6,9,12,15,18,21,None],
              "max_features": [1,2,3,4,5,6,7,8,9,10],
              "min_samples_leaf": [1,2,3,4,5,6,7,8,9,10],
              "criterion": ["gini", "entropy"]}

tree = DecisionTreeClassifier()

tree_cv = RandomizedSearchCV(tree, param_dist, cv = 5)
 
tree_cv.fit(X, Y)

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

Churn=Churn_Data['Churn']
Churn=pd.get_dummies(Churn)
Churn.rename(columns ={0: 'Not_Churn',1: 'Churn'}, inplace = True)
Churn=Churn.sum()
Churn=pd.DataFrame(Churn)

Churn.plot.bar(figsize=(10,5),rot=0, title="Frequency distribution of Churn")
plt.show(block=True)
Y_train = Churn_Data['Churn']
X_train = Churn_Data.drop('Churn', axis=1)

from sklearn.ensemble import RandomForestClassifier
y_imp = Y_train
X_imp  = X_train
rf_model=RandomForestClassifier(bootstrap=True,  max_features = 'sqrt', n_estimators= 30)
rf_model.fit(X_imp,y_imp)

impotance=rf_model.feature_importances_
impotance=pd.DataFrame(impotance)

fetch=X_imp.columns
fetch=pd.DataFrame(fetch)
df_index_ = pd.merge(fetch, impotance, right_index=True, left_index=True)
df_index_

df_index_.plot.bar(x='0_x', y='0_y',figsize=(50,50),fontsize=35,rot=45)
plt.savefig("my_plot.png")

X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=30)


clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
X_test_Pred=clf.predict(X_test)
X_test_Pred

from sklearn.metrics import confusion_matrix
#compute the confusion matrix.
cm = confusion_matrix(y_test,X_test_Pred)
 
#Plot the confusion matrix.
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['Churn','Not Churn'],
            yticklabels=['Churn','Not Churn'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=15)
plt.show()

class_report = classification_report(y_test,X_test_Pred)
print(class_report)

from sklearn.utils import resample
Num_Y=Churn_Data['Churn']
Num_X=Churn_Data.drop(['Churn'],axis=1)

a_train, a_test, b_train, b_test = train_test_split(Num_X, Num_Y, test_size=0.25, random_state=27)

a = pd.concat([a_train, b_train], axis=1)
not_Churn = a[a['Churn']==0]
Churn = a[a['Churn']==1]

not_Churn.shape

Churn.shape

Churn_upsampled = resample(Churn,
                          replace=True,
                          n_samples=len(not_Churn),
                          random_state=27) 

upsampled = pd.concat([not_Churn, Churn_upsampled])
upsampled.Churn.value_counts()


Churn=upsampled['Churn']
Churn=pd.get_dummies(Churn)
Churn.rename(columns ={0: 'Not_Churn',1: 'Churn'}, inplace = True)
Churn=Churn.sum()
Churn=pd.DataFrame(Churn)

Churn.plot.bar(figsize=(10,5),rot=0, title="Frequency distribution of Churn")
plt.show(block=True)

Y_train_ = upsampled.Churn
X_train_ = upsampled.drop('Churn', axis=1)

Y_train = upsampled.Churn
X_train = upsampled.drop('Churn', axis=1)

X_train_, X_test_, y_train_, y_test_ = train_test_split(X_train_, Y_train_, test_size=0.33, random_state=42)

clf = RandomForestClassifier(random_state=0)
clf.fit(X_train_, y_train_)
X_test_=clf.predict(X_test_)
X_test_

cm = confusion_matrix(y_test_,X_test_)

sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['Fraud','Not Fraud'],
            yticklabels=['Fraud','Not Fraud'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=15)
plt.show()

class_report = classification_report(y_test_,X_test_)
print(class_report)

X, y = X_train,Y_train

model = RandomForestClassifier()
n_estimators = [5, 10,20]
max_features = ['sqrt', 'log2']

grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


import pickle
clf = RandomForestClassifier(random_state=0).fit(X_train_, y_train_)
pickle.dump(clf, open('website\churn.pkl', 'wb'))

   




