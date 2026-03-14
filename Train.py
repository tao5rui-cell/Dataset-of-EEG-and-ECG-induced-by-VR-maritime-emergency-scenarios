
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('TKAgg')

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    #'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7, weights='uniform'),
}

for name, model in models.items():
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_validate)
    accuracy = accuracy_score(Y_validate, Y_pred)
    precision = precision_score(Y_validate, Y_pred, average='macro')
    recall = recall_score(Y_validate, Y_pred, average='macro')
    f1 = f1_score(Y_validate, Y_pred, average='macro')
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")






