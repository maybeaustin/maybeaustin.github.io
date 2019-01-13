import numpy as np

from sklearn import datasets, svm
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])

X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()
for k in range(3):
    # We use 'list' to copy, in order to 'pop' later on
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print(scores)  

from sklearn.model_selection import GridSearchCV, cross_val_score
Cs = np.logspace(-6, -1, 10)
clf = GridSearchCV(iid = False, cv = 5, estimator=svc, param_grid=dict(C=Cs),
                   n_jobs=-1)
clf.fit(X_digits[:1000], y_digits[:1000])        

clf.best_score_                                  

clf.best_estimator_.C                            


# Prediction performance on test set is not as good as on train set
clf.score(X_digits[1000:], y_digits[1000:])  




import time

def foo(x):
  return x * x

start = time.time()
I = [i for i in range(0, 100000)]
for i in I:
  foo(i)
end = time.time()
print(end - start)

from multiprocessing import Pool
pool = Pool()
start = time.time()
pool.map(foo, I)
end = time.time()
print(end - start)
pool.close()
