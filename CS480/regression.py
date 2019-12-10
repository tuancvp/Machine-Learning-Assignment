import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


class Regression_Cross_Validation:
	def __init__(self, k=10):
		self.k = k

	def train(self, X_train, Y_train):
		self.X_train = X_train
		self.Y_train = Y_train.flatten()
		print("train: ", self.X_train.shape, self.Y_train.shape)

	def split_data_by_fold(self, num_folds):
		self.num_folds = num_folds
		self.X_train_folds = np.array_split(self.X_train, num_folds)
		self.Y_train_folds = np.array_split(self.Y_train, num_folds)
		# print("split_data_by_fold: ", len(self.X_train_folds[0]), len(self.Y_train_folds))


	def set_data_training(self, X_train_fold, Y_train_fold):
		self.X_train_fold = X_train_fold
		self.Y_train_fold = Y_train_fold


	def loadData(self, fn_data="./regression-dataset/trainInput{}.csv", fn_label="./regression-dataset/trainTarget{}.csv"):
		X_train = []
		Y_train = []

		for file_index in range(1, 11, 1):
			# print("LOAD DATA FROM FILE:{}".format(fn_data.format(file_index)))

			data = pd.read_csv(fn_data.format(file_index), header=None)
			labels = pd.read_csv(fn_label.format(file_index), header=None)

			# print(data.head())
			# print(data.describe())
			# print(data.info())
			# print(len(data), len(labels))
			# print("DATA load from file: {} \nLABELS load from file: {}".format(fn_data.format(file_index), fn_label.format(file_index)))
			data = data.to_numpy(copy=True)
			labels = labels.to_numpy(copy=True)

			if file_index == 1:
				X_train = data
				Y_train = labels
			else:
				X_train = np.vstack((X_train, data))
				Y_train = np.vstack((Y_train, labels))

	
		self.train(X_train, Y_train)

		self.split_data_by_fold(num_folds=10)

		print("DONE LOAD DATE SIZE X: {} y: {}".format(len(X_train), len(Y_train)))



		


	def compute_weight(self, X, y):
		# print("X shape: {} Y shape: {}".format(X.shape, y.shape))
		# weight = []
		# using bias
		# w = (A.T^-1) * b = ( (X.T * X)^-1) * (X.T * y) (5)
		# Building Xbar 
		one = np.ones((X.shape[0], 1))
		Xbar = np.concatenate((one, X), axis = 1)
		# print(Xbar)
		# Calculating weights of the fitting line 
		A = np.dot(Xbar.T, Xbar)
		b = np.dot(Xbar.T, y)
		w = np.dot(np.linalg.pinv(A), b)

		# print('w = ', w)
		
		# # fit the model by Linear Regression
		# regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
		# regr.fit(Xbar, y)

		# # Compare two results
		# print( 'Solution found by scikit-learn  : ', regr.coef_ )

		# print( 'Solution found: ', w.T)

		return w


	def compute_mse(self, X, w, y, lamb):
		one = np.ones((X.shape[0], 1))
		Xbar = np.concatenate((one, X), axis = 1)
		
		y_pred = np.dot(Xbar, w)
		
		# print("y_pred = {}".format(y_pred))
		# print("y_true= {}".format(y))

		# dists = np.reshape(np.sum(y**2, axis=1), (-1, 1) ) + np.sum(y_pred**2, axis=1) - 2 * np.matmul(X, self.X_train_fold.T)
		
		# dists = np.sqrt(dists)
		# Cost Function = mean((yi - xi * w)^2) + penalty term(Regularization): lambda * (mean(W2))
		# in this assignment: Regularization =  0.5λw.T*w
		dists = np.square(y_pred - y).mean() + 0.5 * lamb * np.dot(w, w.T)

		
		# print(dists)

		return dists


	def cross_validation(self):
		lambda_choices = np.arange(0, 4.1, 0.1)

		# print(lambda_choices)
		# return 
		lambda_to_accuracies = {}

		for k in lambda_choices:
			print("process cross_validation k fold k = {}".format(k))
			lambda_to_accuracies[k] = []

			for i in range(self.num_folds):
				X_train_fold = np.concatenate([ fold for j, fold in enumerate(self.X_train_folds) if i != j ] )
				Y_train_fold = np.concatenate([ fold for j, fold in enumerate(self.Y_train_folds) if i != j ] )

				self.set_data_training(X_train_fold, Y_train_fold)
				
				# dists = self.compute_distance(self.X_train_folds[i])
				weight = self.compute_weight(self.X_train_fold, self.Y_train_fold)

				mse = self.compute_mse(self.X_train_folds[i], weight, self.Y_train_folds[i], k)

				# y_pred_fold = self.predict_labels(dists, k = k)

				# print(y_pred)
				# num_correct = np.sum(y_pred_fold == self.Y_train_folds[i])
				# # print(type(self.Y_train_folds[i][0]), type(y_pred_fold[0]))
				# accuracy = float(num_correct) / self.X_train_folds[i].shape[0]
				# # print("accuracy: {}".format(accuracy))
				lambda_to_accuracies[k].append(mse)
				# break
			
			# print(lambda_to_accuracies[k])

			# break
		# print("lambda_to_accuracies: {}".format(lambda_to_accuracies))
		return lambda_choices, lambda_to_accuracies
		# return True

	def process_test_data(self, fn_data="./knn-dataset/testData.csv", fn_label="./knn-dataset/testLabels.csv", k=20):
		data = pd.read_csv(fn_data)
		labels = pd.read_csv(fn_label)
		# print(len(data), len(labels))
		# print("DATA load from file: {} \nLABELS load from file: {}".format(fn_data.format(file_index), fn_label.format(file_index)))
		X_test = data.to_numpy(copy=True)
		Y_test = labels.to_numpy(copy=True)


		dists = self.compute_distance(X_test)

		y_pred_test = self.predict_labels(dists, k=k)
		# print(y_pred)
		num_correct = np.sum(y_pred_test == Y_test)

		# print(type(self.Y_train_folds[i][0]), type(y_pred_fold[0]))

		accuracy = float(num_correct) / X_test.shape[0]

		print("Accuracy on Testset: {}".format(accuracy))
		# print("accuracy: {}".format(accuracy))



	def plotData(self):
		# x, y = zip(*self.X_train)
		x, y = self.X_train.T
		z = self.Y_train
		print("shape X: {} Y: {} Z: {}".format(len(x), len(y), len(z)))
		# plt.scatter(*zip(*self.X_train))
		# plt.scatter(x, y)
		# x, y =
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		# ax.scatter(x, y, z, c = 'b', marker='o')
		ax.scatter(x, y, z)
		# ax.title('Cross-validation on k')
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')
		plt.show()


	def plot_visualization(self, k_choices, k_to_accuracies):
	
		# plot the raw observations
		for k in k_choices:
		# 	accuracies = k_to_accuracies[k]
			print(k_to_accuracies[k])
		# 	plt.scatter([k] * len(accuracies), accuracies)

		# plot the trend line with error bars that correspond to standard deviation
		accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
		accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
		print("accuracies_mean: {}".format(accuracies_mean))
		# for idx, v_mean in enumerate(accuracies_mean):
		# 	print(idx, v_mean)
		# print("accuracies_std: {}".format(accuracies_std))
		# print(accuracies_mean.argmax(), accuracies_mean.max(), len(accuracies_mean))
		
		k_best = accuracies_mean.argmin()

		print("Best k = {} accuracy = {}".format(k_best + 1, accuracies_mean[k_best]))
			
		# self.process_test_data()

		# plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
		# plt.errorbar(k_choices, accuracies_mean)

		# plt.title('Cross-validation on k')
		# plt.xlabel('k')
		# plt.ylabel('Cross-validation accuracy')
		# plt.show()

	def test_SklearnLR(self):
		X_train = self.X_train
		y_train = self.Y_train

		our_weight = self.compute_weight(X_train, y_train)
		print("OUR LinearRegression: {}".format(our_weight))

		lr = LinearRegression(fit_intercept=False)
		lr.fit(X_train, y_train)

		print("LinearRegression no bias w: {}".format(lr.coef_))

		lr = LinearRegression()
		lr.fit(X_train, y_train)

		print("LinearRegression bias: {} w: {}".format(lr.intercept_, lr.coef_))

		rr = Ridge(alpha=0.01) # higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely
		# restricted and in this case linear and ridge regression resembles
		rr.fit(X_train, y_train)
		print("Ridge bias alpha = 0.01 : {} w: {}".format(rr.intercept_, rr.coef_))
		Ridge_train_score = rr.score(X_train,y_train)
		print("Ridge_train_score alpha = 0.01: {}".format(Ridge_train_score))

		rr = Ridge(alpha=100) # higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely
		# restricted and in this case linear and ridge regression resembles
		rr.fit(X_train, y_train)
		print("Ridge alpha = 100 bias: {} w: {}".format(rr.intercept_, rr.coef_))

		Ridge_train_score = rr.score(X_train,y_train)
		print("Ridge_train_score alpha = 100: {}".format(Ridge_train_score))


		lasso = Lasso()
		lasso.fit(X_train,y_train)
		train_score=lasso.score(X_train,y_train)
		coeff_used = np.sum(lasso.coef_!=0)
		print("training score: {}".format(train_score)) 
		print("number of features used: ", coeff_used)
		print("lasso coef: ", lasso.coef_)

		lasso = Lasso(alpha=0.01, max_iter=10e5)
		lasso.fit(X_train,y_train)
		train_score=lasso.score(X_train,y_train)
		coeff_used = np.sum(lasso.coef_!=0)
		print("training score: {}".format(train_score)) 
		print("number of features used: ", coeff_used)
		print("lasso coef: ", lasso.coef_)

k = Regression_Cross_Validation()
k.loadData()
k.test_SklearnLR()
# k.plotData()

# k_choice, k_to_accuracies = k.cross_validation()
# k.plot_visualization(k_choice, k_to_accuracies)
# k.plotData()
