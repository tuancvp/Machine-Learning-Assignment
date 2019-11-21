import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
		# pri


	def print(self):
		print("k = {}".format(self.k))
		print("Len X: {}".format(len(self.X) ))
		print("Len Y: {}".format(len(self.Y) ))

	def loadData(self, fn_data="./regression-dataset/trainInput{}.csv", fn_label="./regression-dataset/trainTarget{}.csv"):
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

	
		self.train(X_train, Y_train)

		self.split_data_by_fold(num_folds=10)

		print("DONE LOAD DATE SIZE X: {} y: {}".format(len(X_train), len(Y_train)))

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
		y_pred = np.zeros(num_test, dtype=np.int64)
		# print(y_pred)
		for i in range(num_test):
			# self.Y_train_fold[np.argsort(dists[i])] sahpe: (990, 1)
			closest_y = self.Y_train_fold[np.argsort(dists[i])][:k]
			# print(dists[i].shape)
			# print(closest_y)
			# print(self.Y_train_fold[np.argsort(dists[i])].shape) # (990, 1)
			# closest_y = closest_y.reshape(-1, 1)

			# print(closest_y, closest_y.shape)
			# print(type(np.bincount(closest_y).argmax()))
			y_pred[i] = np.bincount(closest_y).argmax()

			# print(y_pred[i])
		
		return y_pred




	def cross_validation(self):
		k_choices = np.arange(1, 31, 1)
		# print(k_choices)
		k_to_accuracies = {}
		for k in k_choices:
			print("process cross_validation k fold k = {}".format(k))
			k_to_accuracies[k] = []

			for i in range(self.num_folds):
				X_train_fold = np.concatenate([ fold for j, fold in enumerate(self.X_train_folds) if i != j ] )
				Y_train_fold = np.concatenate([ fold for j, fold in enumerate(self.Y_train_folds) if i != j ] )

				self.set_data_training(X_train_fold, Y_train_fold)
				dists = self.compute_distance(self.X_train_folds[i])
				y_pred_fold = self.predict_labels(dists, k = k)
				# print(y_pred)
				num_correct = np.sum(y_pred_fold == self.Y_train_folds[i])
				# print(type(self.Y_train_folds[i][0]), type(y_pred_fold[0]))
				accuracy = float(num_correct) / self.X_train_folds[i].shape[0]
				# print("accuracy: {}".format(accuracy))
				k_to_accuracies[k].append(accuracy)
				# break
			
			# print(k_to_accuracies[k])

			# break
		return k_choices, k_to_accuracies
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
		# x, y = self.X_train.T

		plt.scatter(*zip(*self.X_train))
		# plt.scatter(x, y)
		# x, y = 
		plt.title('Cross-validation on k')
		plt.xlabel('k')
		plt.ylabel('Cross-validation accuracy')
		plt.show()

k = Regression_Cross_Validation()
k.loadData()
k.plotData()
# k_choices, k_to_accuracies = k.cross_validation()
# k.plot_visualization(k_choices, k_to_accuracies)

# closest_y = np.array([[1], [3], [2] ])
# closest_y = closest_y.flatten()
# print(closest_y)
# print( np.bincount(closest_y))
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
