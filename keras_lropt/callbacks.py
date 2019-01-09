from matplotlib import pyplot as plt
from keras import backend as K
from keras.callbacks import Callback

class LRFinder(Callback):
	"""Finds optimal learning rate"""
	def __init__(self, steps_per_epoch, min_lr=1e-6, max_lr=1e-1):
		super().__init__()
		
		self.steps_per_epoch = steps_per_epoch
		self.min_lr = min_lr
		self.max_lr = max_lr
		
		self.lr_factor = (self.max_lr / self.min_lr) ** (1. / self.steps_per_epoch)
		self.iteration = 0
		self.history = {}

	def on_train_begin(self, logs=None):
		# Begin training, set minimum learning rate
		K.set_value(self.model.optimizer.lr, self.min_lr)


	def on_batch_end(self, batch, logs=None):
		self.iteration += 1
		logs = logs or {}

		# Get current learning rate
		lr = K.get_value(self.model.optimizer.lr)

		# Update learning rate
		lr *= self.lr_factor
		K.set_value(self.model.optimizer.lr, lr)

		# Logs
		self.history.setdefault('lr', []).append(lr)

		for k, v in logs.items():
			self.history.setdefault(k, []).append(v)

	def on_train_end(self, logs=None):
		self.plotLoss()

	def plotLoss(self):
		# Plot loss as function of learning rate (log scale)
		plt.plot(self.history['lr'], self.history['loss'])
		plt.xscale('log')
		plt.xlabel('Learning rate')
		plt.ylabel('Loss')