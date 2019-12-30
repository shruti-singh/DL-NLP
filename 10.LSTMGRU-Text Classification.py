import tensorflow as tf 
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPUs: ", gpus)
if gpus:
	try:
		# Restrict TensorFlow to only use the second GPU
		tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		print("Using GPU:1")
	except RuntimeError as e:
		print(e)

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
from time import time
import random
from tensorboard import version;
print("Tensorboard version: ", version.VERSION)
#load_ext tensorboard
from tensorboard.plugins.hparams import api as hp

data_dir_path = './datasets/'


class LossFunction:
	@staticmethod
	def binary_crossentropy(y_true, y_pred, from_logits=False):
		y_true = tf.cast(y_true, y_pred.dtype)
		def get_epsilon():
			# epsilon_value = 1e-7
			return tf.keras.backend.epsilon()

		if not from_logits:
			if y_pred.op.type == "Sigmoid":
				tf.reduce_mean(tf.math.add(tf.math.negative(tf.math.multiply(y_pred, y_true)), tf.math.log(tf.math.add(1., tf.math.exp(y_pred)))))
			epsilon = get_epsilon()
			clipped_y_pred = tf.clip_by_value(y_pred, clip_value_min=epsilon, clip_value_max=(1.-epsilon))
			bce = tf.math.multiply(y_true, tf.math.log(tf.math.add(clipped_y_pred, epsilon)))
			temp = tf.math.multiply(tf.math.subtract(1., y_true), tf.math.log(tf.math.add(epsilon, tf.math.subtract(1., clipped_y_pred))))
			return tf.math.negative(tf.reduce_mean(tf.math.add(bce, temp)))
		else:
			# - x * z + log(1 + exp(x)), x = logits, z = labels
			return tf.reduce_mean(tf.math.add(tf.math.negative(tf.math.multiply(y_pred, y_true)), tf.math.log(tf.math.add(1., tf.math.exp(y_pred)))))

class Reg:
	@staticmethod
	def l1_reg(weight_matrix):
		return 0.01 * K.sum(K.abs(weight_matrix))
	
	@staticmethod
	def l2_reg(weight_matrix):
		return 0.01 * 0.5 * K.sum(K.square(x))



# Dataset preparation
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
((train_dataset, validation_dataset), test_dataset), info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True, data_dir=data_dir_path, download=False, split=(train_validation_split, tfds.Split.TEST))
encoder = info.features["text"].encoder
print("\n Vocabulary size: ", encoder.vocab_size)

BUFFER_SIZE = 10000
BATCH_SIZE = 64

# fills a buffer with buffer_size elements, then randomly samples elements from this buffer, replacing the selected elements with new elements. For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
# output_shapes returns the shape of each component of an element of this dataset.
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))

validation_dataset = validation_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(validation_dataset))

test_dataset = test_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset))



class MultiLayerLSTM:
	
	def __init__(self, teacher_forcing=False):
		# number of units in 1st and 2nd LSTM layer, and the next dense layer
		self.num_units_l1 = hp.HParam('num_units_l1', hp.Discrete([32, 64]))
		self.num_units_l2 = hp.HParam('num_units_l2', hp.Discrete([16, 32, 64]))
		self.num_units_l3 = hp.HParam('num_units_l3', hp.Discrete([32, 64]))
		self.dropout = hp.HParam('dropout', hp.Discrete([0.3, 0.4]))
		
		#self.learning_rate = hp.HParam('learning_rate', hp.RealInterval(0.01, 0.5))
		self.optimizer = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
		
		self.hparams = {self.optimizer: self.optimizer, self.num_units_l1: self.num_units_l1, self.num_units_l2: self.num_units_l2, self.num_units_l3: self.num_units_l3, self.dropout: self.dropout}
		
		self.teacher_forcing = teacher_forcing
		
		self.model = None
		
		METRIC_ACCURACY = 'accuracy'
		
		self.timestamp = int(time())
		print("MODEL INIT TIME: ", str(self.timestamp))
		self.log_dir = "./tf_logs/lstm_classification_" + str(self.timestamp) +"/"
		with tf.summary.create_file_writer(self.log_dir).as_default():
			hp.hparams_config(hparams=[self.optimizer, self.num_units_l1, self.num_units_l2, self.num_units_l3, self.dropout], metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],)
		
		return
	
	def loss_function(self, y_true, y_pred):
		r = 0.0
		for w in self.model.trainable_weights:
			r += Reg.l1_reg(w)
		l = LossFunction.binary_crossentropy(y_true, y_pred) + r
		return l
	
#	 def loss_function(self, y_true, y_pred):
#		 return tf.keras.losses.binary_crossentropy(y_true, y_pred)
	
	def generate_model(self, params, single_layer=True):
		if single_layer:
			self.model = tf.keras.Sequential()
			self.model.add(tf.keras.layers.Embedding(input_dim=encoder.vocab_size, output_dim=64))
			self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params[self.num_units_l1])))
			self.model.add(tf.keras.layers.Dropout(params[self.dropout]))
			self.model.add(tf.keras.layers.Dense(params[self.num_units_l3], activation='relu'))
			self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
		else:
			self.model = tf.keras.Sequential()
			self.model.add(tf.keras.layers.Embedding(input_dim=encoder.vocab_size, output_dim=64))
			self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params[self.num_units_l1], return_sequences=True)))
			self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params[self.num_units_l2])))
			self.model.add(tf.keras.layers.Dropout(params[self.dropout]))
			self.model.add(tf.keras.layers.Dense(params[self.num_units_l3], activation='relu'))
			self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
		
		print(self.model.summary())
		return self.model
	
	def get_model(self):
		return self.model
	
	def save_model(self):
		cp = int(time())
		model.save_weights(self.logdir + '/saved_models/model_' + cp, save_format='tf')
		return
	
	def compile_model(self, loss_function, optimizer):
		self.model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
		return self.model
	
	def train_model(self, hparams, train_data, cross_validation_data, run_index):
		self.generate_model(hparams)
		self.compile_model(self.loss_function, hparams[self.optimizer])
		
		callbacks = [
			# Early stopping
			tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4),
			tf.keras.callbacks.TensorBoard(log_dir=self.log_dir + run_index),
		]
		
		self.model.fit(train_data, epochs=3, validation_data=cross_validation_data, callbacks=callbacks,verbose=1)
		_, accuracy = self.model.evaluate(cross_validation_data)
		
		return accuracy
	
	def run(self, run_dir, hparams, train_data, cross_validation_data):
		K.clear_session()
		run_index = run_dir.split("-")[1]
		with tf.summary.create_file_writer(run_dir).as_default():
			# record the values used in this trial
			hp.hparams(hparams)
			acc = self.train_model(hparams, train_dataset, validation_dataset, run_index)
			tf.summary.scalar('accuracy', acc, step=int(run_index))
		return acc
	
	def random_search(self, train, cross_val, seed):
		rng = random.Random(seed)
		total_points_explored = 2
		
		acc_params = []
		
		for session_index in range(total_points_explored):
			hparams = {h: h.domain.sample_uniform(rng) for h in self.hparams}
			run_name = "run-%d" % session_index
			print('--- Starting trial: %s' % run_name)
			print({h.name: hparams[h] for h in hparams})
			acc = self.run(self.log_dir + "tune/" + run_name, hparams, train, cross_val)
			session_index += 1
			acc_params.append((acc, hparams))

		#Todo: call setup here only.
		
		return total_points_explored, acc_params
	
	def setup_model(self, hparams, single_layer=True):
		self.generate_model(hparams, single_layer)
		self.compile_model(self.loss_function, hparams[self.optimizer])
		return self.model
	
	def eval_test(self, test):
		_, acc = self.model.evaluate(test)
		print("Acc on tst set: ")
		print(acc)
		return

m = MultiLayerLSTM()
points_explored, acc_params = m.random_search(train_dataset, validation_dataset, 42)
opt_params = sorted(acc_params,key=lambda x: x[0], reverse=True)[0][1]
m.setup_model(opt_params)
m.eval_test(test_dataset)



class GRUNetwork:
	
	def __init__(self, teacher_forcing=False):
		# number of units in 1st and 2nd GRU layer, and the next dense layer
		self.num_units_l1 = hp.HParam('num_units_l1', hp.Discrete([32, 64]))
		self.num_units_l2 = hp.HParam('num_units_l2', hp.Discrete([16, 32, 64]))
		self.num_units_l3 = hp.HParam('num_units_l3', hp.Discrete([32, 64]))
		self.dropout = hp.HParam('dropout', hp.Discrete([0.3, 0.4]))
		
		#self.learning_rate = hp.HParam('learning_rate', hp.RealInterval(0.01, 0.5))
		self.optimizer = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
		
		self.hparams = {self.optimizer: self.optimizer, self.num_units_l1: self.num_units_l1, self.num_units_l2: self.num_units_l2, self.num_units_l3: self.num_units_l3, self.dropout: self.dropout}
		
		self.teacher_forcing = teacher_forcing
		
		self.model = None
		
		METRIC_ACCURACY = 'accuracy'
		
		self.timestamp = int(time())
		self.log_dir = "./tf_logs/gru_classification/" + str(self.timestamp) +"_model/"
		with tf.summary.create_file_writer(self.log_dir).as_default():
			hp.hparams_config(hparams=[self.optimizer, self.num_units_l1, self.num_units_l2, self.num_units_l3, self.dropout], metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],)
		
		return
	
	def loss_function(self, y_true, y_pred):
		r = 0.0
		for w in self.model.trainable_weights:
			r += Reg.l1_reg(w)
		l = LossFunction.binary_crossentropy(y_true, y_pred) + r
		return l
	
#	 def loss_function(self, y_true, y_pred):
#		 return tf.keras.losses.binary_crossentropy(y_true, y_pred)
	
	def generate_model(self, params, single_layer=True):
		if single_layer:
			self.model = tf.keras.Sequential()
			self.model.add(tf.keras.layers.Embedding(input_dim=encoder.vocab_size, output_dim=64))
			self.model.add(tf.keras.layers.GRU(params[self.num_units_l1], recurrent_initializer='glorot_uniform'))
			self.model.add(tf.keras.layers.Dropout(params[self.dropout]))
			self.model.add(tf.keras.layers.Dense(params[self.num_units_l3], activation='relu'))
			self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
		else:
			self.model = tf.keras.Sequential()
			self.model.add(tf.keras.layers.Embedding(input_dim=encoder.vocab_size, output_dim=64))
			self.model.add(tf.keras.layers.GRU(params[self.num_units_l1], recurrent_initializer='glorot_uniform'))
			self.model.add(tf.keras.layers.SimpleRNN(params[self.num_units_l2]))
			self.model.add(tf.keras.layers.Dropout(params[self.dropout]))
			self.model.add(tf.keras.layers.Dense(params[self.num_units_l3], activation='relu'))
			self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
		
		print(self.model.summary())
		return self.model
	
	def get_model(self):
		return self.model
	
	def save_model(self):
		cp = int(time())
		model.save_weights(self.logdir + '/saved_models/model_' + cp, save_format='tf')
		return
	
	def compile_model(self, loss_function, optimizer):
		self.model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
		return self.model
	
	def train_model(self, hparams, train_data, cross_validation_data, run_index):
		self.generate_model(hparams)
		self.compile_model(self.loss_function, hparams[self.optimizer])
		
		callbacks = [
			# Early stopping
			tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4),
			tf.keras.callbacks.TensorBoard(log_dir=self.log_dir + run_index),
		]
		
		self.model.fit(train_data, epochs=10, validation_data=cross_validation_data, callbacks=callbacks,verbose=1)
		_, accuracy = self.model.evaluate(cross_validation_data)
		
		return accuracy
	
	def run(self, run_dir, hparams, train_data, cross_validation_data):
		K.clear_session()
		run_index = run_dir.split("-")[1]
		with tf.summary.create_file_writer(run_dir).as_default():
			# record the values used in this trial
			hp.hparams(hparams)
			acc = self.train_model(hparams, train_dataset, validation_dataset, run_index)
			tf.summary.scalar('accuracy', acc, step=int(run_index))
		return acc
	
	def random_search(self, train, cross_val, seed):
		rng = random.Random(seed)
		total_points_explored = 9
		
		acc_params = []
		
		for session_index in range(total_points_explored):
			hparams = {h: h.domain.sample_uniform(rng) for h in self.hparams}
			run_name = "run-%d" % session_index
			print('--- Starting trial: %s' % run_name)
			print({h.name: hparams[h] for h in hparams})
			acc = self.run(self.log_dir + "tune/" + run_name, hparams, train, cross_val)
			session_index += 1
			acc_params.append((acc, hparams))

		#Todo: call setup here only.
		
		return total_points_explored, acc_params
	
	def setup_model(self, hparams, single_layer=True):
		self.generate_model(hparams, single_layer)
		self.compile_model(self.loss_function, hparams[self.optimizer])
		return self.model
	
	def eval_test(self, test):
		_, acc = self.model.evaluate(test)
		print("Acc on test set: ")
		print(acc)
		return

m = GRUNetwork()
points_explored, acc_params = m.random_search(train_dataset, validation_dataset, 42)
opt_params = sorted(acc_params,key=lambda x: x[0], reverse=True)[0][1]
m.setup_model(opt_params)
m.eval_test(test_dataset)
