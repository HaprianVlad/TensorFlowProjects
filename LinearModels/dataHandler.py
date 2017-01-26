import tempfile
import urllib.request
import pandas as pd
import os
import tensorflow as tf


# DATA LABELS
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

def read_data(): 
	
	print("Data is loading ...")
	
	if os.path.exists("data/train_file.dat"):
   		train_file = open("data/train_file.dat", "r")	
		
	else:
		train_file = open("data/train_file.dat", "w+")
		urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)



	if os.path.exists("data/test_file.dat"):
   		test_file = open("data/test_file.dat", "r")
		
	else:
		test_file = open("data/test_file.dat", "w+")
		urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", test_file.name)
	
	print("Data is transforming ...")	

	

	df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)	
	df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

	
	df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
	df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

	print("Read data done !")	
	
	train_file.close();
	test_file.close();


	return [df_train, df_test]

# CREATE A TENSOR MODEL. This is represented as a dictionary: feature_name -> feature_tensor
def input_fn(df):	
	
	# Creates a dictionary mapping from each continuous feature column name (k) to
	# the values of that column stored in a constant Tensor.
	continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}

	# Creates a dictionary mapping from each categorical feature column name (k)
	# to the values of that column stored in a tf.SparseTensor.
	categorical_cols = {k: tf.SparseTensor(
		indices=[[i, 0] for i in range(df[k].size)],
		values=df[k].values,
		shape=[df[k].size, 1])
		      for k in CATEGORICAL_COLUMNS
	}

	# Merges the two dictionaries into one.
	feature_cols = dict(continuous_cols)
	feature_cols.update(categorical_cols.items())

	# Converts the label column into a constant Tensor.
	label = tf.constant(df[LABEL_COLUMN].values)

	# Returns the feature columns (data matrix X) and the label(y) all represented as tensors.
	return feature_cols, label


def train_input_fn(df_train):
  return input_fn(df_train)

def eval_input_fn(df_test):
  return input_fn(df_test)

# DEFINES THE TRANSFORMATIONS EACH FEATURE_TENSOR WILL SUPPORT.
def feature_transformations():

	## CATEGORICAL FEATURES
	gender  = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])	
	race  = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=["White", "Black"])	
	education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
	relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
	workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
	occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
	native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)

	## CONTINUOS FEATURES
	age = tf.contrib.layers.real_valued_column("age")
	education_num = tf.contrib.layers.real_valued_column("education_num")
	capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
	capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
	hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

	## TRANSFORMATIONS	

	### BUCKETIZATION OF CONTINOUS FEATURES
	age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

	
	## DIFFERENT FEATURE SETS
	wide_columns = [gender, native_country, education, occupation, workclass,
		  relationship, age_buckets,
		  tf.contrib.layers.crossed_column([education, occupation],
		                                   hash_bucket_size=int(1e4)),
		  tf.contrib.layers.crossed_column(
		      [age_buckets, education, occupation],
		      hash_bucket_size=int(1e6)),
		  tf.contrib.layers.crossed_column([native_country, occupation],
		                                   hash_bucket_size=int(1e4))]
	deep_columns = [
		tf.contrib.layers.embedding_column(workclass, dimension=8),
		tf.contrib.layers.embedding_column(education, dimension=8),
		tf.contrib.layers.embedding_column(gender, dimension=8),
		tf.contrib.layers.embedding_column(relationship, dimension=8),
		tf.contrib.layers.embedding_column(native_country,
				                 dimension=8),
		tf.contrib.layers.embedding_column(occupation, dimension=8),
		age,
		education_num,
		capital_gain,
		capital_loss,
		hours_per_week]

	return [wide_columns, deep_columns]


