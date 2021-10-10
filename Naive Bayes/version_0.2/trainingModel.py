# trainingModel.py
import sys

import pandas as pd
import numpy as np

from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import cross_val_score


from constantInit import *
import dataLoad as dl

import preTreatment as pt

np.random.seed(42)

def trainNB():
	passenger_prepared,passenger_labels,test_set,passenger_test = pt.preProcessData()

	sgd_clf = BernoulliNB()
	passenger_labels = passenger_labels.values.reshape((len(passenger_labels.values),))
	sgd_clf.fit(passenger_prepared, passenger_labels)

	print(cross_val_score(sgd_clf, passenger_prepared, passenger_labels, cv=3, scoring="accuracy"))

	predict_data = sgd_clf.predict(passenger_test)
	PassengerIds = test_set['PassengerId']
	results = pd.Series(predict_data,name="Survived",dtype=np.int32)
	submission = pd.concat([PassengerIds,results],axis = 1)

	dl.saveData(submission,'NBClassifier.csv')
