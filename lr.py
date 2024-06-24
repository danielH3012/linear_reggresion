#membuat regresi linear
from __future__ import absolute_import, division,print_function,unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

#load dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')#training
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')#testing
y_train = dftrain.pop('survived')#pop dr dftrain masukin ke y_train
y_eval = dfeval.pop('survived')#sama

CATEGORICAL_COLUMNS = ['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
NUMERIC_COLUMNS = ['age','fare']
feature_columns = []#tempat ngasih makan model buat bikin prediksi
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()#nyari isi datanya ad apa aja(kek set)
  print(vocabulary)
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#input fungsi(bagaimana data akan dipecah kedalam batches dan epochs)
def make_input_fn(data_df,label_df,num_epochs=1000,shuffle=True,batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))#bikin tf.data.Dataset dengan data daan labelnya
    if shuffle:
      ds = ds.shuffle(1000)#ngerandom urutan data
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_fn(dftrain,y_train)
eval_input_fn = make_input_fn(dfeval,y_eval,num_epochs = 1, shuffle = False)

#buat model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

#trainning model
linear_est.train(train_input_fn)#train
result = linear_est.evaluate(eval_input_fn)#get stats dr training data

clear_output()#cls
print(result)

#make prediction
result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[8])
print(y_eval[8])
print('survive probabilities: ', result[8]['probabilities'][1])
print('not survive probabilities: ',result[8]['probabilities'][0])
