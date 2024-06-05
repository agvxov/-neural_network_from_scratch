#!/bin/python3

# Numpy arrays are significantly faster than python lists
#  they also ease matrix operations we will need {multiplication}
import numpy as np
# Pickle is used for object serialization,
#  it will allow us to save our already trained models to disk
import pickle
# We will need to deep copy NN inputs to append a bias
#  without ruining the sample
from copy import deepcopy

def get_xor():
	r = [
		{'in': [0, 0], 'out': [0]},
		{'in': [1, 0], 'out': [1]},
		{'in': [0, 1], 'out': [1]},
		{'in': [1, 1], 'out': [0]},
	]
	return r

# YYY:
#	https://www.kaggle.com/datasets/swaroopmeher/boston-weather-2013-2023
def get_pressure_data():
	import csv
	r = []
	with open('boston_weather_data.csv', mode='r') as file:
		reader = csv.reader(file)
		for row in reader:
			r.append(row[7])
	return r

# Used for transforming a series to I/O batches to the form:
#	{ in: [a], out: a }
# `.in` is always consulted by our network
# `.out` is used for training only, prediction works without it
def batch_data(l : []):
	# 'ino' as in 'arduINO', modelled after their map()
	#   (`map()˙ already un use by a python builtin used for constructing iterators)
	def ino_map(x, old_min, old_max, new_min, new_max):
		return ((x - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
	def normalize(x):   return ino_map(x / (1000 * 2), 0.48, 0.53, 0.0, 1)
	def denormalize(x): return ino_map(x, 0.0, 1, 0.48, 0.53) * (1000 * 2)
	# number of elements used for constructing a single batch
	BATCH_SIZE = 8
	d = []
	for i in [l[i:i + BATCH_SIZE] for i in range(1, len(l), BATCH_SIZE)]:
		try:
			h = dict()
			h['in']  = [normalize(float(f)) for f in i[:BATCH_SIZE-1]]
			h['out'] = normalize(float(i[BATCH_SIZE-1]))
			d.append(h)
		except Exception as e: pass #print(e)
	return {'data': d, 'normalize': normalize, 'denormalize': denormalize}

# Activation function which could be swapped out arbitrarily
def sigmoid_activation(x): return 1/(1+np.exp(-x))
# Derivative of sigmoid_activation(x)
#  NOTE: the real deactivation would be `sigmoid_activation(x) * (1 - sigmoid_activation(x))`
#         however, we will only be passing values which are already activated!
def sigmoid_deactivation(a): return a * (1 - a)

class NN:
	learning_rate = 0
	# Architecture:
	# {
	#   ARCHITECTURE := [3, 4, 4, 1]
	#    O X O X O \
	#    O X O X O - O
	#    O X O X O /
	#      \ O X O /
	#   // This will results in the following weight matrix shapes:
	#		(4, 3), (4, 4), (4, 1)
	#   // NOTE: notice how one layer is "missing",
	#             as we dont need weights for the input layer
	# }
	architecture = None
	# [Layer]
	layers = None
	# Each layer will be stored as its own object.
	# The reason for this is that:
	#  Storing each neuron individually would keep us from using
	#   numpy matrix operations on layers.
	#   While speed is not a top priority here, it would lead to much more
	#   verbose and confusing code.
	#   And, due to Python's primitive/reference type differentiation
	#   we would also have trouble with using for loops.
	#  Storing layers without a container class would result in a bunch of
	#   parallel arrays. Now, this could be benefitial for performance
	#   in a low level environment when interfacing with the GPU directly,
	#   but we are writting Python...
	#   Anyways, it should go without saying that parellel arrays are ugly.
	class Layer:
		def __init__(self, input_size : int, neuron_count : int, normalizer : int):
			self.weights      = np.random.rand(input_size, neuron_count)
			# Normalizing the weights (based on the neuron count) is 
			#  something ive seen in an example.
			#  Being a hyper-parameter, its hard to say whether it actually
			#  helps or not. What I can tell is that in simple problems
			#  {xor} it is counter productive.
			self.weights      = self.weights / np.sqrt(normalizer)
			self.activation   = sigmoid_activation
			self.deactivation = sigmoid_deactivation
			# Most sources save the activations, which is fundemantelly the same
			#  as what we are doing since the activations of this layer is
			#  the input of the next. However due to the input layer being "virtual",
			#  we are left with a choice either way, we either store the activations
			#  and shoehorn in the initial input or store the inputs and shoehorn
			#  in the last activation. With such a layer based program architecture
			#  i find the later more elegant as the last activation is also the overall
			#  output of the network and having a `prediction` variable does not hurt
			#  code quality: https://commadot.com/wtf-per-minute/
			self.inputs       = None
			self.deltas       = None	# buffer change values as calculated by  backtracking
		def __str__(self): # for printing our network architecture
			r  = f"     \033[35m{self.weights.shape}\033[0m\n{self.weights}\n"
			return r
		def calculate_deltas(self, error : np.array):
			# Notice how this is not a dot product,
			#  we actually want element-wise multiplication
			#  ie. weighting by the error
			return error * self.deactivation(self.inputs)
		def predict(self):
			return self.activation(
						np.dot(self.inputs, self.weights)
					)

	def __init__(self, architecture : [], learning_rate : int):
		self.architecture  = architecture
		self.learning_rate = learning_rate

		self.layers = []
		# The neuron count of the previous layer tells us
		#  the input size, this gets combined with the current
		# NOTE: `+ 1` is always the addition of a bias input node
		for i in np.arange(0, len(architecture) - 2):
			l = self.Layer(architecture[i] + 1, architecture[i+1] + 1, architecture[i])
			self.layers.append(l)
		# The last layer does not get a bias
		l = self.Layer(architecture[-2] + 1, architecture[-1], architecture[-2])
		self.layers.append(l)
	def __str__(self): # also for printing
		r = f"\033[1;34mNeural Network @ {id(self)}:\033[0m"
		for i, l in enumerate(self.layers):
			r += f"\n\033[34m--- Layer {i}:\033[0m\n"
			r += str(l)
		r += "\n\033[1;34m###\033[0m"
		return r


	# Boring serialization stuff
	def save(self):
		from datetime import datetime
		with open(str(datetime.now()).replace(' ', '=') + ".pkl", 'wb') as f:
			pickle.dump(self, f)
	@staticmethod
	def load(id_ : str): # Constructor from file
		with open(id_ + ".pkl", 'rb') as f:
			o = pickle.load(f)
		return o

	# This internal function is used both by ˙train()˙ and `predict()`
	def predict_(self, data : np.ndarray):
		self.layers[0].inputs = np.atleast_2d(data)
		for previous_layer, current_layer in [(self.layers[li-1], self.layers[li]) for li in range(1, len(self.layers))]:
			# NOTE: `.predict()` feeds from ˙.inputs`
			#self.layers[li].inputs = self.layers[li-1].predict()
			current_layer.inputs = previous_layer.predict()
		return self.layers[-1].predict()

	def train(self, data, epochs):
		def train_(data : {}):
			# We do so called 'online learning' where the weights are adjusted after
			#  each input (in contrast to calculating the loss for the entire dataset)
			prediction = self.predict_(data['in'])
			# For every layer we will work out the error in reverse order.
			# NOTE: some sources do `prediction - data['out']` which is equivalent,
			#        as long as we dont confuse the operands
			delta_output_sum = -(data['out'] - prediction)
			self.layers[-1].deltas = delta_output_sum * self.layers[0].deactivation(prediction)

			# We wish to iterate in reverse order so we create a nifty "alias"
			rl = self.layers[::-1]
			# For ever consequent layer we utalize the deltas of the previous
			for previous_layer, current_layer in [(rl[li], rl[li+1]) for li in range(len(rl)-1)]:
				error = np.dot(previous_layer.deltas, previous_layer.weights.T)
				current_layer.deltas = previous_layer.calculate_deltas(error)
			# We update the weights
			for l in self.layers:
				l.weights -= self.learning_rate * np.dot(l.inputs.T, l.deltas)
		# Unlike the error calculations we use to update the weights,
		#  this function operates on the while dataset and serves
		#  as an overview of learning
		def loss_function(data : [], targets : []):
			targets     = np.atleast_2d(targets)
			predictions = np.empty_like(targets)
			for i, d in enumerate(data):
				predictions[0][i] = self.predict_(d)
			return .5 * np.sum((predictions - targets) ** 2)

		# Appending input layer bias
		data_buffer = deepcopy(data)
		for d in data_buffer: d['in'] = np.append(d['in'], 1)

		for epoch in range(epochs):
			for d in data_buffer: train_(d)
			if epoch % 100 == 0:
				#loss = loss_function(d['in'], d['out'])
				loss = loss_function([d['in'] for d in data_buffer], [d['out'] for d in data_buffer])
				print(f"[INFO] epoch={epoch}, loss={loss}")

	def predict(self, data : []):
		p = np.append(data, 1)
		return self.predict_(p)

# -----

def xor():
	data = get_xor()
	# Simpler architectures have an easier time learning easier problems,
	#  requiring less epochs
	# Conceptulize it as the model "overthinking" the problem
	#n = NN([2, 2, 4, 2, 3, 1], .5)
	n = NN([2, 2, 1], .5)
	print(n)

	n.train(data, epochs=2000)
	print(n)

	for d in data:
		out = round(n.predict(d['in'])[0][0], 3)
		print(f"In: {d['in']}\nOut: {out}\nActual: {d['out'][0]}\n")

def pressure():
	samples = batch_data(get_pressure_data())

	PARTITIONING_INDEX = int(len(samples['data']) * .75)
	training_data = samples['data'][:PARTITIONING_INDEX]
	checking_data = samples['data'][PARTITIONING_INDEX:]

	# This architecture seems to be doing slightly better than
	#  something more bloated such as `NN([7, 8, 8, 1], .5)`
	# and significantly better than something longer
	#  such as NN([7, 4, 4, 4, 1], .5)
	n = NN([7, 4, 4, 1], .5)
	#n = NN.load('2024-06-05=10:56:28.122959')
	print(n)

	n.train(training_data, epochs=5000)
	print(n)
	#n.save()

	# This makes our denormalization valid as a matrix operation
	vd = np.vectorize(samples['denormalize'])
	for d in checking_data:
		out           = n.predict(d['in'])
		out_actual    = vd(round(out[0][0], 3))
		in_actual     = vd(d['in'])
		actual_actual = vd(d['out'])
		print(f"In: {in_actual}\nOut: {out_actual}\nActual: {actual_actual}\n")

if __name__ == '__main__':
	#xor()
	pressure()
