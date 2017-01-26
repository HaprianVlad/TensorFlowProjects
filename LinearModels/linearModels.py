import tensorflow as tf

import tempfile
import argparse
import sys
import warnings

from dataHandler import read_data
from dataHandler import train_input_fn
from dataHandler import eval_input_fn
from dataHandler import feature_transformations

def runLearning(model_type, train_steps):
	# READ THE RAW DATA
	[train_data, test_data] = read_data()


	# DEFINE TRANSFORMATIONS FOR THE TENSOR FEATURES IN TRAIN DATA
	[wide_columns, deep_columns] = feature_transformations()

	# DEFINE THE LINEAR MODEL. BIAS COLUMN ADDED AUTOMATICALLY.
	model_dir = tempfile.mkdtemp()
	

	# TRAIN THE MODEL. NOTICE THAT HERE WE PASS train_input_fn(df_train), WHICH WILL CONSTRUCT OUR TENSOR MODEL. THE FEATURE TRANSFORMATIONS WILL BE APPLIED ON THIS MODEL
	print("Training the model ...")
	if (model_type == "wide"):
		m = tf.contrib.learn.LinearClassifier(
			feature_columns=wide_columns, 
			model_dir=model_dir, 
			optimizer=tf.train.FtrlOptimizer(
				learning_rate=0.1,
				l1_regularization_strength=1.0,
				l2_regularization_strength=1.0
				)
			)
		

	if (model_type == "deep"):
		m = tf.contrib.learn.DNNClassifier(feature_columns=deep_columns, model_dir=model_dir, hidden_units=[100, 50])
		
	
	if (model_type == "wide_n_deep"):
		m = tf.contrib.learn.DNNLinearCombinedClassifier(
			linear_feature_columns=wide_columns,
		        dnn_feature_columns=deep_columns,
		        dnn_hidden_units=[100, 50], 
			model_dir=model_dir)

	m.fit(input_fn=lambda:train_input_fn(train_data), steps=train_steps)
		

	# TEST THE MODEL
	print("Testing the model ...")
	results = m.evaluate(input_fn=lambda:eval_input_fn(test_data), steps=1)
	for key in sorted(results):
	    print (str(key) + " : " + str(results[key]))

FLAGS = None

def main(_):
	runLearning(FLAGS.model_type, FLAGS.train_steps)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.register("type", "bool", lambda v: v.lower() == "true")

	parser.add_argument("--model_type",type=str,default="wide", help="Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
	parser.add_argument("--train_steps", type=int, default=200, help="Number of training steps.")
	
	FLAGS, unparsed = parser.parse_known_args()
	tf.logging.set_verbosity(tf.logging.ERROR)
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
