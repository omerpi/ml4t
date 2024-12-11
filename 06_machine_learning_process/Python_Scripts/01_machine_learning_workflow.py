#!/usr/bin/env python
# coding: utf-8

# # Basic Walk-Through: k-nearest Neighbors

# This notebook contains several examples that illustrate the machine learning workflow using a dataset of house prices.
# 
# We will use the fairly straightforward [k-nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html) (KNN) algorithm that allows us to tackle both regression and classification problems.
# 
# In its default sklearn implementation, it identifies the k nearest data points (based on the Euclidean distance) to make a prediction. It predicts the most frequent class among the neighbors or the average outcome in the classification or regression case, respectively.

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.neighbors import (KNeighborsClassifier, 
                               KNeighborsRegressor)
from sklearn.model_selection import (cross_val_score, 
                                     cross_val_predict, 
                                     GridSearchCV)
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler, scale
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from yellowbrick.model_selection import ValidationCurve, LearningCurve


# In[3]:


sns.set_style('whitegrid')


# ## Get the Data

# ### Kings County Housing Data

# Data from [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction)
# 
# Download via API:
# 
# ```kaggle datasets download -d harlfoxem/housesalesprediction```

# In[4]:


DATA_PATH = Path('..', 'data')


# In[5]:


house_sales = pd.read_csv('kc_house_data.csv')
house_sales = house_sales.drop(
    ['id', 'zipcode', 'lat', 'long', 'date'], axis=1)
house_sales.info()


# ## Select & Transform Features

# ### Asset Prices often have long tails

# In[6]:


sns.distplot(house_sales.price)
sns.despine()
plt.tight_layout();


# ### Use log-transform

# Useful for dealing with [skewed data](http://onlinestatbook.com/2/transformations/log.html).

# In[7]:


X_all = house_sales.drop('price', axis=1)
y = np.log(house_sales.price)


# ### Mutual information regression

# See sklearn [docs](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection). Covered later in Chapter 6 of the book.

# In[8]:


mi_reg = pd.Series(mutual_info_regression(X_all, y),
                   index=X_all.columns).sort_values(ascending=False)
mi_reg


# In[9]:


X = X_all.loc[:, mi_reg.iloc[:10].index]


# ### Bivariate Scatter Plots

# In[10]:


g = sns.pairplot(X.assign(price=y), y_vars=['price'], x_vars=X.columns)
sns.despine();


# ### Explore Correlations

# In[11]:


X.info()


# In[12]:


correl = X.apply(lambda x: spearmanr(x, y)[0])
correl.sort_values().plot.barh();


# ## KNN Regression

# ### In-sample performance with default settings

# KNN uses distance to make predictions; it requires standardization of variables to avoid undue influence based on scale

# In[13]:


X_scaled = scale(X)


# In[14]:


model = KNeighborsRegressor()
model.fit(X=X_scaled, y=y)


# In[15]:


y_pred = model.predict(X_scaled)


# ### Regression Error Metrics

# In[16]:


from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             mean_squared_log_error, 
                             median_absolute_error, 
                             explained_variance_score, 
                             r2_score)


# #### Computing the prediction error

# The error is the deviation from the true value, whereas a residual is the deviation from an estimated value, e.g., an estimate of the population mean.

# In[17]:


error = (y - y_pred).rename('Prediction Errors')


# In[18]:


scores = dict(
    rmse=np.sqrt(mean_squared_error(y_true=y, y_pred=y_pred)),
    rmsle=np.sqrt(mean_squared_log_error(y_true=y, y_pred=y_pred)),
    mean_ae=mean_absolute_error(y_true=y, y_pred=y_pred),
    median_ae=median_absolute_error(y_true=y, y_pred=y_pred),
    r2score=explained_variance_score(y_true=y, y_pred=y_pred)
)


# In[19]:


fig, axes = plt.subplots(ncols=3, figsize=(15, 4))
sns.scatterplot(x=y, y=y_pred, ax=axes[0])
axes[0].set_xlabel('Log Price')
axes[0].set_ylabel('Predictions')
axes[0].set_ylim(11, 16)
axes[0].set_title('Predicted vs. Actuals')
sns.distplot(error, ax=axes[1])
axes[1].set_title('Residuals')
pd.Series(scores).plot.barh(ax=axes[2], title='Error Metrics')
fig.suptitle('In-Sample Regression Errors', fontsize=16)
sns.despine()
fig.tight_layout()
fig.subplots_adjust(top=.88)


# ### Cross-Validation

# Manual hyperparameter tuning; using [Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline-chaining-estimators) ensures proper scaling for each fold using train metrics to standardize test data.

# In[20]:


def rmse(y_true, pred):
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=pred))

rmse_score = make_scorer(rmse)


# In[21]:


cv_rmse = {}
n_neighbors = [1] + list(range(5, 51, 5))
for n in n_neighbors:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('knn', KNeighborsRegressor(n_neighbors=n))])                    
    cv_rmse[n] = cross_val_score(pipe, 
                                X=X, 
                                y=y,
                                scoring=rmse_score,
                                cv=5)


# In[22]:


cv_rmse = pd.DataFrame.from_dict(cv_rmse, orient='index')
best_n, best_rmse = cv_rmse.mean(1).idxmin(), cv_rmse.mean(1).min()
cv_rmse = cv_rmse.stack().reset_index()
cv_rmse.columns =['n', 'fold', 'RMSE']


# In[23]:


ax = sns.lineplot(x='n', y='RMSE', data=cv_rmse)
ax.set_title(f'Cross-Validation Results KNN | Best N: {best_n:d} | Best RMSE: {best_rmse:.2f}');


# #### Actuals vs Predicted

# In[24]:


pipe = Pipeline([('scaler', StandardScaler()),
                 ('knn', KNeighborsRegressor(n_neighbors=best_n))])
y_pred = cross_val_predict(pipe, X, y, cv=5)

ax = sns.scatterplot(x=y, y=y_pred)
y_range = list(range(int(y.min() + 1), int(y.max() + 1)))
pd.Series(y_range, index=y_range).plot(ax=ax, lw=2, c='darkred');


# #### Cross-Validation Errors

# In[25]:


error = (y - y_pred).rename('Prediction Errors')


# In[26]:


scores = dict(
    rmse=np.sqrt(mean_squared_error(y_true=y, y_pred=y_pred)),
    rmsle=np.sqrt(mean_squared_log_error(y_true=y, y_pred=y_pred)),
    mean_ae=mean_absolute_error(y_true=y, y_pred=y_pred),
    median_ae=median_absolute_error(y_true=y, y_pred=y_pred),
    r2score=explained_variance_score(y_true=y, y_pred=y_pred)
)


# In[27]:


fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
sns.scatterplot(x=y, y=y_pred, ax=axes[0])
axes[0].set_xlabel('Log Price')
axes[0].set_ylabel('Predictions')
axes[0].set_ylim(11, 16)
sns.distplot(error, ax=axes[1])
pd.Series(scores).plot.barh(ax=axes[2], title='Error Metrics')
fig.suptitle('Cross-Validation Regression Errors', fontsize=24)
fig.tight_layout()
plt.subplots_adjust(top=.8);


# ### GridSearchCV with Pipeline

# See sklearn [docs](https://scikit-learn.org/stable/modules/grid_search.html#tuning-the-hyper-parameters-of-an-estimator).
# 

# In[28]:


pipe = Pipeline([('scaler', StandardScaler()), 
                 ('knn', KNeighborsRegressor())])

n_folds = 5
n_neighbors = tuple(range(5, 101, 5))

param_grid = {'knn__n_neighbors': n_neighbors}

estimator = GridSearchCV(estimator=pipe,
                         param_grid=param_grid,
                         cv=n_folds,
                         scoring=rmse_score,
#                          n_jobs=-1
                        )
estimator.fit(X=X, y=y)


# In[29]:


cv_results = estimator.cv_results_


# In[30]:


test_scores = pd.DataFrame({fold: cv_results[f'split{fold}_test_score'] for fold in range(n_folds)}, 
                           index=n_neighbors).stack().reset_index()
test_scores.columns = ['k', 'fold', 'RMSE']


# In[31]:


mean_rmse = test_scores.groupby('k').RMSE.mean()
best_k, best_score = mean_rmse.idxmin(), mean_rmse.min()


# In[32]:


sns.pointplot(x='k', y='RMSE', data=test_scores, scale=.3, join=False, errwidth=2)
plt.title('Cross Validation Results')
sns.despine()
plt.tight_layout()
plt.gcf().set_size_inches(10, 5);


# ### Train & Validation Curves mit yellowbricks
# 

# See background on [learning curves](https://en.wikipedia.org/wiki/Learning_curve) and yellowbrick [docs](https://www.scikit-yb.org/en/latest/).

# In[33]:


fig, ax = plt.subplots(figsize=(16, 9))
val_curve = ValidationCurve(KNeighborsRegressor(),
                      param_name='n_neighbors',
                      param_range=n_neighbors,
                      cv=5,
                      scoring=rmse_score,
#                       n_jobs=-1,
                      ax=ax)
val_curve.fit(X, y)
val_curve.poof()
sns.despine()
fig.tight_layout();


# In[34]:


fig, ax = plt.subplots(figsize=(16, 9))
l_curve = LearningCurve(KNeighborsRegressor(n_neighbors=best_k), 
                        train_sizes=np.arange(.1, 1.01, .1),
                        scoring=rmse_score, 
                        cv=5, 
#                         n_jobs=-1,
                        ax=ax)
l_curve.fit(X, y)
l_curve.poof()
sns.despine()
fig.tight_layout();


# ## Binary Classification

# In[35]:


y_binary = (y>y.median()).astype(int)


# In[36]:


n_neighbors = tuple(range(5, 151, 10))
n_folds = 5
scoring = 'roc_auc'


# In[37]:


pipe = Pipeline([('scaler', StandardScaler()), 
                 ('knn', KNeighborsClassifier())])

param_grid = {'knn__n_neighbors': n_neighbors}

estimator = GridSearchCV(estimator=pipe,
                         param_grid=param_grid,
                         cv=n_folds,
                         scoring=scoring,
#                          n_jobs=-1
                        )
estimator.fit(X=X, y=y_binary)


# In[38]:


best_k = estimator.best_params_['knn__n_neighbors']


# In[39]:


fig, ax = plt.subplots(figsize=(16, 9))
val_curve = ValidationCurve(KNeighborsClassifier(),
                      param_name='n_neighbors',
                      param_range=n_neighbors,
                      cv=n_folds,
                      scoring=scoring,
#                       n_jobs=-1,
                      ax=ax)
val_curve.fit(X, y_binary)
val_curve.poof()
sns.despine()
fig.tight_layout();


# In[40]:


fig, ax = plt.subplots(figsize=(16, 9))
l_curve = LearningCurve(KNeighborsClassifier(n_neighbors=best_k), 
                        train_sizes=np.arange(.1, 1.01, .1),
                        scoring=scoring, 
                        cv=5, 
#                         n_jobs=-1,
                        ax=ax)
l_curve.fit(X, y_binary)
l_curve.poof()
sns.despine()
fig.tight_layout();


# ### Classification Metrics

# See sklearn [docs](https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation-quantifying-the-quality-of-predictions) for details.

# In[41]:


from sklearn.metrics import (classification_report,
                             accuracy_score,
                             zero_one_loss,
                             roc_auc_score,
                             roc_curve,
                             brier_score_loss,
                             cohen_kappa_score,
                             confusion_matrix,
                             fbeta_score,
                             hamming_loss,
                             hinge_loss,
                             jaccard_score,
                             log_loss,
                             matthews_corrcoef,
                             f1_score,
                             average_precision_score,
                             precision_recall_curve)


# |Name| API|
# |:----|----|
# |Area Under the Receiver Operating Characteristic Curve (ROC AUC) | roc_auc_score(y_true, y_score[, …])|
# |Receiver operating characteristic (ROC) | roc_curve(y_true, y_score[, …])|
# |Average precision (AP) | average_precision_score(y_true, y_score)|
# |Precision-recall pairs | precision_recall_curve(y_true, …)|
# |Precision, recall, F-measure and support | precision_recall_fscore_support(…)|
# |F1 Score | f1_score(y_true, y_pred[, labels, …])|
# |F-beta Score | fbeta_score(y_true, y_pred, beta[, …])|
# |Precision | precision_score(y_true, y_pred[, …]|
# |Recall | recall_score(y_true, y_pred[, …])|
# |Main classification metrics | classification_report(y_true, y_pred)|
# |confusion matrix | confusion_matrix(y_true, y_pred[, …])|
# |Accuracy classification score | accuracy_score(y_true, y_pred)|
# |Zero-one classification loss | zero_one_loss(y_true, y_pred[, …])|
# |Average Hamming loss | hamming_loss(y_true, y_pred[, …])|
# |Brier score | brier_score_loss(y_true, y_prob[, …])|
# |Cohen’s kappa | cohen_kappa_score(y1, y2[, labels, …])|
# |Average hinge loss | hinge_loss(y_true, pred_decision[, …])|
# |Jaccard similarity coefficient | jaccard_similarity_score(y_true, y_pred)|
# |Log loss, aka logistic loss or cross-entropy loss | log_loss(y_true, y_pred[, eps, …])|
# |Matthews correlation coefficient (MCC) | matthews_corrcoef(y_true, y_pred[, …])|

# In[42]:


y_score = cross_val_predict(KNeighborsClassifier(best_k), 
                           X=X, 
                           y=y_binary, 
                           cv=5, 
                           n_jobs=-1, 
                           method='predict_proba')[:, 1]


# #### Using Predicted Probabilities

# In[43]:


pred_scores = dict(y_true=y_binary,y_score=y_score)


# ##### ROC AUC

# In[44]:


roc_auc_score(**pred_scores)


# In[45]:


cols = ['False Positive Rate', 'True Positive Rate', 'threshold']
roc = pd.DataFrame(dict(zip(cols, roc_curve(**pred_scores))))


# ##### Precision-Recall

# In[46]:


precision, recall, ts = precision_recall_curve(y_true=y_binary, probas_pred=y_score)
pr_curve = pd.DataFrame({'Precision': precision, 'Recall': recall})


# ##### F1 Score - Optimize Threshold

# In[47]:


f1 = pd.Series({t: f1_score(y_true=y_binary, y_pred=y_score>t) for t in ts})
best_threshold = f1.idxmax()


# ##### Plot

# In[48]:


roc.info()


# In[49]:


fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

ax = sns.scatterplot(x='False Positive Rate', y='True Positive Rate', data=roc, size=5, legend=False, ax=axes[0])
axes[0].plot(np.linspace(0,1,100), np.linspace(0,1,100), color='k', ls='--', lw=1)
axes[0].fill_between(y1=roc['True Positive Rate'], x=roc['False Positive Rate'], alpha=.5,color='darkred')
axes[0].set_title('Receiver Operating Characteristic')


sns.scatterplot(x='Recall', y='Precision', data=pr_curve, ax=axes[1])
axes[1].set_ylim(0,1)
axes[1].set_title('Precision-Recall Curve')


f1.plot(ax=axes[2], title='F1 Scores', ylim=(0,1))
axes[2].set_xlabel('Threshold')
axes[2].axvline(best_threshold, lw=1, ls='--', color='k')
axes[2].text(s=f'Max F1 @ {best_threshold:.2f}', x=.5, y=.95)
sns.despine()
fig.tight_layout();


# ##### Average Precision

# In[50]:


average_precision_score(y_true=y_binary, y_score=y_score)


# ##### Brier Score

# In[51]:


brier_score_loss(y_true=y_binary, y_prob=y_score)


# #### Use Predictions after thresholding

# In[52]:


y_pred = y_score > best_threshold


# In[53]:


scores = dict(y_true=y_binary, y_pred=y_pred)


# ##### F-beta Score

# In[54]:


fbeta_score(**scores, beta=1)


# In[55]:


print(classification_report(**scores))


# ##### Confusion Matrix

# In[56]:


confusion_matrix(**scores)


# ##### Accuracy

# In[57]:


accuracy_score(**scores)


# ##### Zero-One Loss

# In[58]:


zero_one_loss(**scores)


# ##### Hamming Loss

# Fraction of labels that are incorrectly predicted

# In[59]:


hamming_loss(**scores)


# ##### Cohen's Kappa

# Score that expresses the level of agreement between two annotators on a classification problem.

# In[60]:


cohen_kappa_score(y1=y_binary, y2=y_pred)


# ##### Hinge Loss

# In[61]:


hinge_loss(y_true=y_binary, pred_decision=y_pred)


# ##### Jaccard Similarity

# In[62]:


jaccard_score(**scores)


# ##### Log Loss / Cross Entropy Loss

# In[63]:


log_loss(**scores)


# ##### Matthews Correlation Coefficient

# In[64]:


matthews_corrcoef(**scores)


# ## Multi-Class

# In[65]:


y_multi = pd.qcut(y, q=3, labels=[0,1,2])


# In[66]:


n_neighbors = tuple(range(5, 151, 10))
n_folds = 5
scoring = 'accuracy'


# In[67]:


pipe = Pipeline([('scaler', StandardScaler()), 
                 ('knn', KNeighborsClassifier())])

param_grid = {'knn__n_neighbors': n_neighbors}

estimator = GridSearchCV(estimator=pipe,
                         param_grid=param_grid,
                         cv=n_folds,
                         n_jobs=-1
                        )
estimator.fit(X=X, y=y_multi)


# In[68]:


y_pred = cross_val_predict(estimator.best_estimator_, 
                           X=X, 
                           y=y_multi, 
                           cv=5, 
                           n_jobs=-1, 
                           method='predict')


# In[69]:


print(classification_report(y_true=y_multi, y_pred=y_pred))

