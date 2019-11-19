import pandas as pd
import numpy as np

class knn:
	def __init__(self, k):
		self.k = k
		# self.X_train
		# self.Y_train
		# 

	def train(self, X_train, Y_train):
		self.X_train = X_train
		self.Y_train = Y_train
		print("train: ", self.X_train.shape, self.Y_train.shape)

	def split_data_by_fold(self, num_folds):
		print("split_data_by_fold")
		self.num_folds = num_folds
		self.X_train_folds = np.array_split(self.X_train, num_folds)
		self.Y_train_folds = np.array_split(self.Y_train, num_folds)
		print("split_data_by_fold: ", len(self.X_train_folds[0]), len(self.Y_train_folds))


	def set_data_training(self, X_train_fold, Y_train_fold):
		self.X_train_fold = X_train_fold
		self.Y_train_fold = Y_train_fold
		# pri


	def print(self):
		print("k = {}".format(self.k))
		print("Len X: {}".format(len(self.X) ))
		print("Len Y: {}".format(len(self.Y) ))

	def loadData(self, fn_data="data{}.csv", fn_label="labels{}.csv"):
		X_train = []
		Y_train = []

		for file_index in range(1, 11, 1):
			data = pd.read_csv(fn_data.format(file_index))
			labels = pd.read_csv(fn_label.format(file_index))
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

			
			# self.X_train = np.vstack((self.X_train, data.to_numpy(copy=True)))
			# self.Y_train = np.vstack((self.Y_train, labels.to_numpy(copy=True)))

		# print(len(X_train), len(Y_train), type(X_train), type(Y_train))
		
		self.train(X_train, Y_train)
		self.split_data_by_fold(num_folds=10)

		# print("DONE LOAD DATE SIZE X: {} y: {}".format(len(X_train), len(Y_train)))

	def compute_distance(self, X):
		# Compute the l2 distance between all test points and all training point
		
		# print("shape X: {} X_train_fold {}".format(X.shape, self.X_train_fold.shape))

		# shape X: (110, 64) X_train_fold (990, 64)
		# expand equation (x-y)^2 = x^2 + y^2 - 2xy
		# np.sum(X**2, axis=1), (-1, 1) shape: (110, 1) [2D] - sample: X = np.array([[2], [3]])
		# np.sum(self.X_train_fold**2, axis=1) shape: (990, ) [1D] - sample: Y = np.array([3, 4, 5])
		# X + Y = [[5 6 7] [6 7 8]]
		# dist =  (110, 990) - (110, 64) * (64, 990)

		dists = np.reshape(np.sum(X**2, axis=1), (-1, 1) ) + np.sum(self.X_train_fold**2, axis=1) - 2 * np.matmul(X, self.X_train_fold.T)
		dists = np.sqrt(dists)
		return dists

	def predict_labels(self, dists, k):

		"""
		Given a matrix of distances between test points and training points,
		predict a label for each test point.
		Inputs:
		- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
		  gives the distance betwen the ith test point and the jth training point.
		Returns:
		- y: A numpy array of shape (num_test,) containing predicted labels for the
		  test data, where y[i] is the predicted label for the test point X[i].  
		"""

		num_test = dists.shape[0]
		y_pred = np.zeros(num_test)
		# print(y_pred)
		for i in range(num_test):
			# self.Y_train_fold[np.argsort(dists[i])] sahpe: (990, 1)
			closest_y = self.Y_train_fold[np.argsort(dists[i])][:k]
			# print(dists[i].shape)
			# print(self.Y_train_fold[np.argsort(dists[i])].shape) # (990, 1)
			print(closest_y, closest_y.shape)
			print(np.bincount(closest_y))
			y_pred[i] = np.bincount(closest_y).argmax()
		
		return y_pred




	def cross_validation(self):
		k_choices = np.arange(1, 31, 1)
		print(k_choices)
		k_to_accuracies = {}
		for k in k_choices:
			k_to_accuracies[k] = {}
			for i in range(self.num_folds):
				X_train_fold = np.concatenate([ fold for j, fold in enumerate(self.X_train_folds) if i != j ] )
				Y_train_fold = np.concatenate([ fold for j, fold in enumerate(self.Y_train_folds) if i != j ] )

				self.set_data_training(X_train_fold, Y_train_fold)
				dists = self.compute_distance(self.X_train_folds[i])
				y_pred = self.predict_labels(dists, k = 10)
				print(y_pred)
				
				break
			
			break




		# return True


k = knn(10)
k.loadData()
k.cross_validation()
# X = np.array([[2], [3]])
# Y = np.array([3, 4, 5])
# print(X + Y)
# print(X.shape, Y.shape)
# X = np.array([ [[2, 3], [3, 5], [3, 2]], [[2, 3], [3, 5], [3, 2]], [[2, 3], [3, 5], [3, 2]] ])
# print(np.reshape(np.array([2, 3]), (-1, 1)))
# print(np.sum( X**2, axis=1))
# X = np.array([1, 2, 3, 4])
# print(np.reshape(np.array([2, 3]), (-1, 1)) +X)
# print(np.reshape(np.array([2, 3]), (-1, 1)) + np.array[3, 4])
# k.print()
# X = np.array( [ [ [2, 4], [3, 5] ], [[2, 4], [3, 5] ], [[2, 4], [3, 5] ], [[2, 4], [3, 5] ] ]  )
# # # print(X ** X)
# X_sum = np.sum(X**2, axis=0) # [22 38]
# print(X_sum)

# X_sum = np.sum(X**2, axis=1) # [13 34 13]
# print(X_sum)

# X_sum = np.sum(X**2, axis=2) # [13 34 13]
# print(X_sum)


# X_sum = np.reshape(X_sum, (-1, 1))

# print(X_sum)
