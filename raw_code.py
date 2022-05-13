#!/usr/bin/env python
# coding: utf-8

# # Indian Liver Patient Dataset
# 
# ### The following notebook contains the following
# 
# ### • Exploratory analysis of the dataset 
# ### • Data cleansing (one-hot encoding, filling missing values)
# ### • Predictive model training for benchmark score
# ### • Testing data-preprocessing methods on benchmark score (feature selection, min-max scaling)
# ### • Implementing only the methods which increase the score
# ### • SMOTE oversampling
# ### • Hyperparameter tuning
# ### • Final model evaluation and rebalancing tests
# 
# #### approx run time: 350 seconds

# In[1]:


#import requried libraries and modules

# !pip install optuna -- not used as of submission
# !pip install plotly
# !pip install scikit-plot
# !pip install shap
# !pip install mlxtend

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.datasets import make_classification
from sklearn.model_selection import HalvingGridSearchCV
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import mlxtend
from mlxtend.plotting import plot_decision_regions
import warnings
import shap
import optuna
import scikitplot as skplt
warnings.filterwarnings("ignore")
np.random.seed(0)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read in the data and set column headers

colnames=['Age', 'Sex', 'TB Total Bilirubin', 'DB Direct Bilirubin', 'Alkphos Alkaline Phosphotase',
          'Sgpt Alamine Aminotransferase','Sgot Aspartate Aminotransferase',
         'TP Total Proteins', 'ALB Albumin', 'A/G Ratio Albumin and Globulin Ratio','Target'] 

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
                 names = colnames, header=None)


# ### Exploratory analysis of the dataset and feature space

# In[3]:


#inspect the dataframe

df.head()


# In[4]:


#data types and feature value counts
df.info()


# In[5]:


#basic distribution info
df.describe()


# In[6]:


#target feature imbalance
sns.countplot(data=df, x = 'Target')


# In[7]:


#count target feature imbalance 
Counter(df['Target'])


# In[8]:


sns.countplot(data=df, x = 'Sex',palette = 'Blues')


# In[9]:


df.Age.hist()


# In[10]:


import matplotlib.gridspec as gridspec
fig = plt.figure(constrained_layout=True, figsize=(16, 12))
grid = gridspec.GridSpec(ncols=6, nrows=3, figure=fig)
ax5 = fig.add_subplot(grid[2, :4])

ax5.set_title('Age Distribution')

sns.distplot(df[['Age']],
                 hist_kws={
                 'rwidth': 1,
                 'edgecolor': 'black',
                 'alpha': 0.8},
                 color = '#4682B4')

plt.show()


# In[11]:


ax = sns.countplot(x="Target", hue="Sex", data=df)


# In[12]:


df['Sex'].replace('Female',0,inplace=True)
df['Sex'].replace('Male',1,inplace=True)


# In[13]:


sns.scatterplot(data=df, x="TB Total Bilirubin", y="DB Direct Bilirubin")


# In[14]:


# visualize all relationships between features before calculating correlations
sns.pairplot(df)


# In[15]:


sns.pairplot(df,hue='Target')


# In[16]:


#plot outliers
fig, ax = plt.subplots(figsize=(15,8))
sns.boxplot(data=df, width = 0.5, ax=ax, fliersize=3)
plt.xticks(rotation=45)
plt.title("Outliers")


# In[17]:


corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True,annot=True)


# In[18]:


df.drop('Target', axis=1).corrwith(df.Target).plot(kind='bar', grid=True, figsize=(12, 8), 
                                                   title="Correlation with target")


# In[19]:


plt.figure(figsize=(12,6))
palette=['lightskyblue','orangered']
s1=sns.boxenplot(x=df.Sex, y=df.Age, hue=df.Target, palette=palette, linewidth=3)
handles = s1.get_legend_handles_labels()[0]
s1.legend(handles, ['No disease', 'Disease'])
s1.set_title("Sex and age for target",fontsize=16)
plt.show()


# In[20]:


plt.figure(figsize=(12,4))
labels = ['Female','Male']
sns.countplot(df['Sex'], hue=df['Target'], palette=['lightblue','orangered'], saturation=0.8)
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Target count in genders', fontsize=16)
plt.legend(loc='upper right', fontsize=16, labels=['No disease', 'Disease'])
plt.show()


# #### There are some linearly correlated features, such as Total and Direct Bilirubin. These will likely be removed by RFE after finding benchmark scores for comparison. There may be little benefit removing them at this stage.

# In[21]:


from mpl_toolkits.mplot3d import Axes3D

feature1 = df['TP Total Proteins'].values
feature2 = df['ALB Albumin'].values 
feature3 = df['DB Direct Bilirubin'].values

df['Target']=df['Target'].astype('str')

c = df['Target'].values
df['Target']=df['Target'].astype('int')
c[c=='1'] = 'b' #negative diagnosis diabetes
c[c=='2'] = 'r' #positive diagnosis diabetes

fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(feature1, feature2, feature3, c=c)
ax.set_xlabel('Bilirubin')
ax.set_ylabel('Protein')
ax.set_zlabel('Albumin')

plt.show()


# In[22]:


feature4 = df['Alkphos Alkaline Phosphotase'].values
feature5 = df['Sgpt Alamine Aminotransferase'].values 
feature6 = df['Sgot Aspartate Aminotransferase'].values

df['Target']=df['Target'].astype('str')

c = df['Target'].values
df['Target']=df['Target'].astype('int')
c[c=='1'] = 'b' #negative diagnosis diabetes
c[c=='2'] = 'r' #positive diagnosis diabetes

fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(feature4, feature5, feature6, c=c)
ax.set_xlabel('AAP')
ax.set_ylabel('S AL AT')
ax.set_zlabel('S AS AT')

plt.show()


# In[23]:


import plotly.express as px

# 3D scatterplot:

fig = px.scatter_3d(df, x='TB Total Bilirubin', y='Alkphos Alkaline Phosphotase', z='Age', size='ALB Albumin',
              color='Sex', opacity=0.8)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()


# Outliers exist in this space.

# In[24]:


#check for null values
df.isna().sum()


# In[25]:


df['A/G Ratio Albumin and Globulin Ratio'].mean()


# In[26]:


df['A/G Ratio Albumin and Globulin Ratio'].median()


# In[27]:


#replace missing values with feature median
df['A/G Ratio Albumin and Globulin Ratio'].fillna(df['A/G Ratio Albumin and Globulin Ratio'].median(),inplace=True)


# In[28]:


df['A/G Ratio Albumin and Globulin Ratio'].isna().sum()


# In[29]:


#one-hot encoding sex
df = pd.get_dummies(df, columns=['Sex'])


# In[30]:


df


# In[31]:


#generate baseline variables to assess accuracy without feature selection
X = df.drop('Target',axis=1)
y = df['Target']


# ### Prevent leakage by obtaining a hold-out

# In[32]:


X, X_holdout, y, y_holdout = train_test_split(X, y, test_size=0.2,
                                              random_state=111,shuffle=True)


# ### Feature importance

# In[33]:


#identify low-value features ---- optional: this can take up to 5 extra minutes ----
# estimator = SVR(kernel="linear")
# selector = RFE(estimator, n_features_to_select=5, step=1)
# selector = selector.fit(X, y)
# print(selector.ranking_)


# In[34]:


#visualise feature importance
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X, y)
importances = sel.estimator_.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X.shape[1]),['Age','TB','DB','ALP','ALT','AST','TP','ALB','A/G','Sex1','Sex2'])
plt.xlim([-1, X.shape[1]])
plt.show()


# In[35]:


importances


# ### Obtaining benchmark scores before further processing via cross-validation on training set

# In[36]:


from sklearn import svm

# models
models = []
models.append(('KNN',KNeighborsClassifier()))
models.append(('RF',RandomForestClassifier()))
models.append(('LR',LogisticRegression(max_iter=2500)))
models.append(('SVM',svm.SVC()))

results = []
names = []
scoring = 'accuracy'

# cross validation
for name, model in models:
    kfold = model_selection.KFold(n_splits=5,random_state = 111,shuffle=True)
    cv_results = model_selection.cross_val_score(model, X,y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[37]:


y_holdout.value_counts()


# In[38]:


# parameter range
parameter_range = np.arange(1, 10, 1)

# 10-fold cross validation
train_score, test_score = validation_curve(KNeighborsClassifier(), X, y,
                                           param_name = "n_neighbors",
                                           param_range = parameter_range,
                                           cv = 5, scoring = "accuracy")

# mean and standard deviation (training)
mean_train_score = np.mean(train_score, axis = 1)
std_train_score = np.std(train_score, axis = 1)

# mean and standard deviation (test)
mean_test_score = np.mean(test_score, axis = 1)
std_test_score = np.std(test_score, axis = 1)

# mean accuracy (train and test)
plt.plot(parameter_range, mean_train_score,
         label = "Training Accuracy", color = 'b')
plt.plot(parameter_range, mean_test_score,
label = "Cross Validation Accuracy", color = 'g')
plt.fill_between(parameter_range, mean_train_score - std_train_score, mean_train_score + std_train_score, color="gray")
plt.fill_between(parameter_range, mean_test_score - std_test_score, mean_test_score + std_test_score, color="gainsboro")
# Creating the plot
plt.title("Validation Curve with KNN Classifier")
plt.xlabel("Number of Neighbours")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc = 'best')
plt.show()


 


# In[39]:


parameter_range = np.arange(1,50,1)
train_score, test_score = validation_curve(RandomForestClassifier(), X, y,
                                           param_name = "n_estimators",
                                           param_range = parameter_range,
                                           cv = 5, scoring = "accuracy")

# mean and standard deviation (train)
mean_train_score = np.mean(train_score, axis = 1)
std_train_score = np.std(train_score, axis = 1)

# mean and standard deviation (test)
mean_test_score = np.mean(test_score, axis = 1)
std_test_score = np.std(test_score, axis = 1)

# Plot mean accuracy (train and test)
plt.plot(parameter_range, mean_train_score,
         label = "Training Accuracy", color = 'b')
plt.plot(parameter_range, mean_test_score,
label = "Cross Validation Accuracy", color = 'g')
plt.fill_between(parameter_range, mean_train_score - std_train_score, mean_train_score + std_train_score, color="gray")
plt.fill_between(parameter_range, mean_test_score - std_test_score, mean_test_score + std_test_score, color="gainsboro")
# Creating the plot
plt.title("Validation Curve with RF Classifier")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc = 'best')
plt.show()


# In[40]:


# Removing redundant features
X = X.iloc[:,[0,1,3,4,5,6]]


# In[41]:


results_post_feature_selection = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=5, random_state = 111,shuffle=True)
    cv_results_post_feature_selection = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results_post_feature_selection.append(cv_results_post_feature_selection)
    msg = "%s: %f (%f)" % (name, cv_results_post_feature_selection.mean(), cv_results_post_feature_selection.std())
    print(msg)


# In[42]:


for name, model in models:
    kfold = model_selection.KFold(n_splits=5, random_state = 111,shuffle=True)
    cv_results_post_feature_selection = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results_post_feature_selection.append(cv_results_post_feature_selection)
    print(f'{name} A total improvement of: {cv_results_post_feature_selection.mean() - cv_results.mean()}')


# ### Scaling the data and moving on with only models of interest

# In[43]:


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled)
X_holdout = X_holdout.iloc[:,[0,1,3,4,5,6]]
# X_holdout = scaler.transform(X_holdout)
X_holdout[:] = scaler.transform(X_holdout)


# In[44]:


Models = []
Models.append(('KNN',KNeighborsClassifier()))
Models.append(('RF',RandomForestClassifier()))

results_post_feature_selection = []
for name, model in Models:
    kfold = model_selection.KFold(n_splits=5, random_state = 111,shuffle=True)
    cv_results_post_feature_selection = model_selection.cross_val_score(model, X_scaled, y, cv=kfold, scoring=scoring)
    results_post_feature_selection.append(cv_results_post_feature_selection)
    msg = "%s: %f (%f)" % (name, cv_results_post_feature_selection.mean(), cv_results_post_feature_selection.std())
    print(msg)


# In[45]:


X_standard = StandardScaler().fit_transform(X)


# In[46]:


results_post_feature_selection = []
for name, model in Models:
    kfold = model_selection.KFold(n_splits=5, random_state = 111,shuffle=True)
    cv_results_post_feature_selection = model_selection.cross_val_score(model, X_standard, y, cv=kfold, scoring=scoring)
    results_post_feature_selection.append(cv_results_post_feature_selection)
    msg = "%s: %f (%f)" % (name, cv_results_post_feature_selection.mean(), cv_results_post_feature_selection.std())
    print(msg)


# ### Oversampling
# #### Reduces bias if model is to be used on unseen data by increasing recall

# In[47]:


sm = SMOTE(random_state=42)
Counter(y)


# In[48]:


X_res, y_res = sm.fit_resample(X_scaled, y)
Counter(y_res)


# In[49]:


results_smote = []
for name, model in Models:
    kfold = model_selection.KFold(n_splits=10, random_state = 111,shuffle=True)
    cv_results_post_feature_selection = model_selection.cross_val_score(model, X_res, y_res, cv=kfold, scoring=scoring)
    results_post_feature_selection.append(cv_results_post_feature_selection)
    msg = "%s: %f (%f)" % (name, cv_results_post_feature_selection.mean(), cv_results_post_feature_selection.std())
    print(msg)


# #### Pre-balanced fit causes the model to 'cheat' and predict a certain class for every attempt

# In[50]:


from sklearn import metrics
model = RandomForestClassifier()
model.fit(X,y)
y_pred = model.predict(X_holdout)
print(classification_report(y_holdout,y_pred))
model = KNeighborsClassifier()
model.fit(X,y)
y_pred = model.predict(X_holdout)
print(classification_report(y_holdout,y_pred))


# In[51]:


# baseline performance pre-tuning
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
for name, model in Models:
    model.fit(X,y)
scoring = ['precision_macro', 'recall_macro','accuracy']
for model in Models:
    print(model)
for name, model in Models:
    scores = cross_val_score(model, X_holdout, y_holdout, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# ### Final baseline classification accuracy scores:
# #### Random Forest - 0.68
# #### KNN - 0.63

# In[52]:


model = RandomForestClassifier()
model.fit(X_res,y_res)
y_pred = model.predict(X_holdout)
print(classification_report(y_holdout,y_pred))
model = KNeighborsClassifier()
model.fit(X_res,y_res)
y_pred = model.predict(X_holdout)
print(classification_report(y_holdout,y_pred))


# ### Hyperparameter tuning

# #### Random Forest

# In[82]:


#optuna first (not used because its bayesian and i just want nice plots)

def objective(trial):
      
      n_estimators = trial.suggest_int('n_estimators', 100, 120)
      max_depth = int(trial.suggest_loguniform('max_depth', 1, 10))
#       criterion = trial.suggest_categorical('criterion', 'entropy', 'gini')
      min_samples_leaf = trial.suggest_int('min_samples_leaf', 1,5)
      clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,min_samples_leaf=min_samples_leaf)
      return cross_val_score(clf, X_res, y_res, 
           n_jobs=-1, cv=3).mean()


# In[83]:


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=600)
trial = study.best_trial
print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))


# In[84]:


optuna.visualization.plot_optimization_history(study)


# In[85]:


optuna.visualization.plot_slice(study)


# In[86]:


optuna.visualization.plot_param_importances(study)


# In[98]:


from sklearn.metrics import f1_score
parameters = {'n_estimators': [100, 120, 150, 500, 700, 1000],
              'criterion': ['entropy', 'gini'], 
              'max_depth': [None,1,3,5,7,9],
              'min_samples_leaf': [1,3,5]}

clf = RandomForestClassifier(random_state=111)
gs_rf = GridSearchCV(clf, 
                     parameters, 
                     cv=5, 
                     n_jobs=-1,verbose=10,scoring = 'roc_auc')

gs_rf.fit(X_res, y_res)
print('Best auroc: %.3f' % gs_rf.best_score_)
print('\nBest params:\n', gs_rf.best_params_)
rf_best = gs_rf.best_params_


# #### KNN

# In[ ]:


def objective(trial):
      
      n_neighbors = trial.suggest_int('n_neighbors', 1, 6)
      leaf_size = int(trial.suggest_loguniform('leaf_size', 25, 35))
#       

#       weights = trial.suggest_categorical('weights','uniform','distance')
      clf = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size)
      return cross_val_score(clf, X_res, y_res, 
           n_jobs=-1, cv=3).mean()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=600)
trial = study.best_trial
print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))


# In[ ]:


optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_slice(study)


# In[ ]:


optuna.visualization.plot_param_importances(study)


# In[99]:


parameters = {'n_neighbors' : [1,2,3,4,5,6],
             'weights' : ['uniform','distance'],
             'algorithm' : ['auto','ball_tree','brute'],
              'p' : [1,2],
             'leaf_size' : [25,30,35]}

clf = KNeighborsClassifier()
gs_knn = GridSearchCV(clf, 
                     parameters, 
                     cv=5, 
                     n_jobs=-1,verbose=10, scoring = 'roc_auc')

gs_knn.fit(X_res, y_res)
print('Best accuracy: %.3f' % gs_knn.best_score_)
print('\nBest params:\n', gs_knn.best_params_)
knn_best = gs_knn.best_params_


# #### Double checking the parameter range wasn't too small

# In[100]:



parameter_range = np.arange(1,30,1)
train_score, test_score = validation_curve(RandomForestClassifier(), X_res, y_res,
                                           param_name = "max_depth",
                                           param_range = parameter_range,
                                           cv = 10, scoring = "accuracy")

# mean and standard deviation (training)
mean_train = np.mean(train_score, axis = 1)
std_train = np.std(train_score, axis = 1)

# mean and standard deviation (testing)
mean_test = np.mean(test_score, axis = 1)
std_test = np.std(test_score, axis = 1)

# Plot mean accuracy (train and test)
plt.plot(parameter_range, mean_train,
         label = "Training Accuracy", color = 'b')
plt.plot(parameter_range, mean_test,
label = "Cross Validation Accuracy", color = 'g')
plt.fill_between(parameter_range, mean_train - std_train, mean_train + std_train, color="gray")
plt.fill_between(parameter_range, mean_test - std_test, mean_test + std_test, color="gainsboro")
# Creating the plot
plt.title("Validation Curve with RF Classifier")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc = 'best')
plt.show()


# In[101]:


parameter_range = np.arange(1,45,1)
train_score, test_score = validation_curve(KNeighborsClassifier(), X_res, y_res,
                                           param_name = "leaf_size",
                                           param_range = parameter_range,
                                           cv = 10, scoring = "accuracy")

# mean and standard deviation (training)
mean_train = np.mean(train_score, axis = 1)
std_train = np.std(train_score, axis = 1)

# mean and standard deviation (testing)
mean_test = np.mean(test_score, axis = 1)
std_test = np.std(test_score, axis = 1)

# mean accuracy (training and testing)
plt.plot(parameter_range, mean_train,
         label = "Training Accuracy", color = 'b')
plt.plot(parameter_range, mean_test,
label = "Cross Validation Accuracy", color = 'g')
plt.fill_between(parameter_range, mean_train - std_train, mean_train + std_train, color="gray")
plt.fill_between(parameter_range, mean_test - std_test, mean_test + std_test, color="gainsboro")
# plot
plt.title("Validation Curve with k-NN Classifier")
plt.xlabel("leaf_size")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc = 'best')
plt.show()


# In[102]:


Models_tuned = []
Models_tuned.append(('KNN',KNeighborsClassifier(**knn_best)))
Models_tuned.append(('RF',RandomForestClassifier(**rf_best)))


# In[103]:


rf_best


# ## Model evaluation 

# ### Random Forest

# In[104]:


from sklearn.metrics import roc_curve
from sklearn import metrics


# In[105]:


classifier = RandomForestClassifier(**rf_best)
tpr_list = []
auc_list = []
cv = model_selection.KFold(n_splits=10,random_state = 111,shuffle=True)

mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X_res, y_res):
    probas_ = classifier.fit(X_res, y_res).predict_proba(X_holdout)
    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_holdout, probas_[:, 1],pos_label=2)
    tpr_list.append(np.interp(mean_fpr, fpr, tpr))
    tpr_list[-1][0] = 0.0
    roc_auc = metrics.auc(fpr, tpr)
    auc_list.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tpr_list, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(auc_list)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tpr_list, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=16)
plt.ylabel('True Positive Rate',fontsize=16)
plt.title('Cross-Validation ROC of Random Forest',fontsize=15)
plt.legend(loc="lower right", prop={'size': 14})
plt.show()
print(mean_auc)


# In[106]:


model = RandomForestClassifier(**rf_best)
model.fit(X_res, y_res)
y_probas = model.predict_proba(X_holdout)
skplt.metrics.plot_roc(y_holdout, y_probas)

plt.show()


# In[107]:


model.score(X_holdout,y_holdout)


# In[108]:


y_pred = model.predict(X_holdout)

print(classification_report(y_holdout, y_pred))


# In[109]:


skplt.metrics.plot_precision_recall(y_holdout, y_probas)

plt.show()


# In[110]:


tn, fp, fn, tp = confusion_matrix(list(y_holdout), list(y_pred), labels=[0, 1]).ravel()
accuracy_score(y_holdout, y_pred)


# In[111]:


cf_matrix = confusion_matrix(y_holdout, y_pred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');


ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

plt.show()


# In[112]:


model.fit(X_res,y_res)
scoring = ['precision_macro', 'recall_macro','accuracy']

scores = cross_validate(model, X_holdout, y_holdout, scoring=scoring)
sorted(scores.keys())
['fit_time', 'score_time', 'test_precision_macro', 'test_recall_macro']
scores['test_recall_macro']
print(scores)


# In[113]:


#log loss
from sklearn.metrics import log_loss
log_loss(y_holdout,y_probas)


# In[114]:


#MAE
from sklearn.metrics import mean_absolute_error

mean_absolute_error(
    y_holdout,
    y_pred
)


#  ## K-nearest neighbors

# In[115]:


classifier = KNeighborsClassifier(**knn_best)
tpr_list = []
auc_list = []
cv = model_selection.KFold(n_splits=10,random_state = 111,shuffle=True)

mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X_res, y_res):
    probas_ = classifier.fit(X_res, y_res).predict_proba(X_holdout)
    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_holdout, probas_[:, 1],pos_label=2)
    tpr_list.append(np.interp(mean_fpr, fpr, tpr))
    tpr_list[-1][0] = 0.0
    roc_auc = metrics.auc(fpr, tpr)
    auc_list.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tpr_list, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(auc_list)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tpr_list, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=16)
plt.ylabel('True Positive Rate',fontsize=16)
plt.title('Cross-Validation ROC of Random Forest',fontsize=15)
plt.legend(loc="lower right", prop={'size': 14})
plt.show()
print(mean_auc)


# In[116]:


model2 = KNeighborsClassifier(**knn_best)
model2.fit(X_res, y_res)
y_probas = model2.predict_proba(X_holdout)
skplt.metrics.plot_roc(y_holdout, y_probas)

plt.show()


# In[117]:


y_pred2 = model2.predict(X_holdout)
print(classification_report(y_holdout, y_pred2))


# In[118]:


cf_matrix = confusion_matrix(y_holdout, y_pred2)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

plt.show()


# In[119]:


skplt.metrics.plot_precision_recall(y_holdout, y_probas)

plt.show()


# In[120]:


#MAE
from sklearn.metrics import mean_absolute_error

mean_absolute_error(
    y_holdout,
    y_pred2
)


# In[121]:


clf = KNeighborsClassifier(**knn_best)
clf.fit(X_res.iloc[:,:2],y_res)
plt.xlim(0, 0.8)
plt.ylim(0, 0.6)
plot_decision_regions(X_res.iloc[:,:2].values, y_res.values, clf=clf, zoom_factor=10)
plt.show()


# In[122]:


clf = RandomForestClassifier(**rf_best)
clf.fit(X_res.iloc[:,:2],y_res)
plt.xlim(0.4, 0.6)
plt.ylim(0, 0.3)
plot_decision_regions(X=X_res.iloc[:,:2].values, y=y_res.values, clf=clf, zoom_factor=100000)
plt.show()


# #### Follow-up cross-validation 

# In[123]:


print(Models_tuned)
for name, model in Models_tuned:

    scores = cross_val_score(model, X_holdout, y_holdout, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[124]:


#log loss
from sklearn.metrics import log_loss
log_loss(y_holdout,y_probas)


# In[125]:



my_model_rf = RandomForestClassifier(**rf_best).fit(X_res, y_res)
explainer = shap.TreeExplainer(my_model_rf)
shap_values_rf = explainer.shap_values(X_holdout,y_holdout)
shap.summary_plot(shap_values_rf[1],X_res, plot_type="bar")


# In[126]:


my_model_knn = KNeighborsClassifier(**knn_best).fit(X_res, y_res)
explainer = shap.KernelExplainer(my_model_knn.predict,X_res)
shap_values_knn = explainer.shap_values(X_holdout)
shap.summary_plot(shap_values_knn,X_res, plot_type="bar")


# In[127]:


shap.summary_plot(shap_values_rf, X_holdout)


# In[128]:


shap.summary_plot(shap_values_knn, X_holdout)


# In[129]:


shap.initjs()
shap.force_plot(explainer.expected_value, shap_values_rf[1], X_holdout)


# In[130]:


shap.dependence_plot("Age", shap_values_rf[0], X_holdout)


# In[131]:


shap.dependence_plot("TB Total Bilirubin", shap_values_rf[0], X_holdout)


# ### Supplimentary data balance tests

# In[ ]:


# type(y_holdout)


# In[ ]:


# pd.set_option('display.max_rows', 100)
# y_holdout


# In[ ]:


# type(X_holdout)


# In[ ]:


# y_holdout.value_counts()


# In[ ]:


# mask = (y_holdout < 2)
# idx, = np.where(mask)
# y_holdout_balanced = y_holdout.drop(y_holdout.index[idx[:len(idx)//2]])


# In[ ]:


# y_holdout_balanced.value_counts()


# In[ ]:


# X_holdout = pd.DataFrame(X_holdout)
# X_holdout_balanced = X_holdout[X_holdout.index.isin(y_holdout_balanced.index)]


# #### Random Forest

# In[ ]:


# model = RandomForestClassifier(**study.best_trial.params)
# model.fit(X_res, y_res)
# model.score(X_holdout,y_holdout)


# In[ ]:


# model = RandomForestClassifier(**study.best_trial.params)
# model.fit(X_res, y_res)
# model.score(X_holdout_balanced,y_holdout_balanced)


# In[ ]:


# y_pred = model.predict(X_holdout)

# print(classification_report(y_holdout, y_pred))


# In[ ]:


### how does the model perform on balanced data?
# y_pred = model.predict(X_holdout_balanced)
# print(classification_report(y_holdout_balanced,y_pred))


# #### KNN

# In[ ]:


# model = KNeighborsClassifier(**study2.best_trial.params)
# model.fit(X_res, y_res)
# model.score(X_holdout,y_holdout)


# In[ ]:


# model = KNeighborsClassifier(**study2.best_trial.params)
# model.fit(X_res, y_res)
# model.score(X_holdout_balanced,y_holdout_balanced)


# In[ ]:


# y_pred = model.predict(X_holdout)
# print(classification_report(y_holdout,y_pred))


# In[ ]:


# y_pred = model.predict(X_holdout_balanced)
# print(classification_report(y_holdout_balanced,y_pred))


# Conclusion: A single model approach would yield a more efficient process and result

# In[ ]:


# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Activation,Dropout
# from tensorflow.keras.models import Model


# In[ ]:


# ip_layer = Input(shape=(X.shape[1],))
# dl1 = Dense(100, activation='relu')(ip_layer)
# dl2 = Dense(50, activation='relu')(dl1)
# dl3 = Dense(25, activation='relu')(dl2)
# dl4 = Dense(10, activation='relu')(dl3)
# output = Dense(1)(dl4)


# In[ ]:


# model = Model(inputs = ip_layer, outputs=output)
# model.compile(loss="mean_absolute_error" , optimizer="adam", metrics=["mean_absolute_error"])


# In[ ]:


# pip install graphviz


# In[ ]:


# pip install pydot


# In[ ]:


# import graphviz
# import pydot


# In[ ]:


# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# In[ ]:


# history = model.fit(X_res, y_res, batch_size=5, epochs=500, verbose=1, validation_split=0.2)


# In[ ]:


# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train','test'], loc='upper left')
# plt.show()


# In[ ]:


# y_pred = model.predict(X_holdout)


# In[ ]:


# from sklearn import metrics
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_holdout, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_holdout, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_holdout, y_pred)))


# In[ ]:





# In[ ]:




