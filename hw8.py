# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 23:31:03 2017

@author: Ashwini
"""


from nltk import tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import tensorflow as tf
import glob

import os
import random

SavedModelFile = "SaveModel" #to use in test mode
model_path = '/Python/HW8'
TrainMode = False  # for first time set it True to train it, False: for test

#data_folder = r"C:\Python\HW8\input"
data_folder = '/Python/HW8/input'
files = sorted(glob.glob(os.path.join(data_folder, "*.txt")))
chapters=[]
for fn in files:
    with open(fn,encoding="utf-8") as f:
        chapters.append(f.read().replace('\n', ' '))
Merged_Text = ' '.join(chapters)
#statement = re.split(r' *[\\?!][\'"\)\]]* *', Merged_Text)

sentences = tokenize.sent_tokenize(Merged_Text)

#removing \n and convetion sentence in lower case
step1=[item.replace('\n', '') for item in sentences]
step2 = [x.lower() for x in step1]


vectorizer = CountVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(step2)


Tokens = [(w, vectors.getcol(i).sum()) for w, i in vectorizer.vocabulary_.items()]



#getting most frequent 250 features
category = []
for i, w in Tokens:
    item = (i,w)
    category.append(item)
#sorted from the most frequent word
category = sorted(category, key = lambda x: -x[1])

#Creating bag of words 
features = []
for i in range(250):
    features.append(category[i][0])

    
def processSentence(s):
    Vector = np.zeros(250)
    for word in s.split(' '):
        if word in features:
            Vector[features.index(word)] += 1
    return Vector
                   
Train_Data_Vector = []
Train_Sentences = []

for s in step2:
    V = processSentence(s)
    if not all(V == 0):
        Train_Data_Vector.append(V)
        Train_Sentences.append(s)


#Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 300
display_step = 1

#Network Parameters
n_hidden_1 = 250
n_hidden_2 = 125
n_hidden_3 = 50
n_input = 250

#input
X = tf.placeholder("float", [None, n_input])

weights = {
           'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
           'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
           'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
           'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
           'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
           'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
          'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
          'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
          'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
          'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
          'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
          'decoder_b3': tf.Variable(tf.random_normal([ n_input])),
}

#Building the encoder 
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))

    return layer_3
#Building the decoder 
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    return layer_3
    

#Construct model
encoder_output = encoder(X)
decoder_output = decoder(encoder_output)
    
#prediction
y_pred = decoder_output
#Targets are the input data
y_true = X

#Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#Initializing the variables
init = tf.global_variables_initializer()

#Training Model
if TrainMode:
    sess = tf.InteractiveSession()
    sess.run(init)
    
    savepoint = tf.train.Saver()
    
    total_batch = int(len(Train_Data_Vector)/batch_size)
    
    for epoch in range(training_epochs):
        j = 0
        for i in range(total_batch):
            batch_xs = Train_Data_Vector[j:j + batch_size]
            _, c = sess.run([optimizer, cost], feed_dict={X:batch_xs})
            j= j+batch_size
            
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))
        
     #saving trained model        
    savepoint.save(sess, os.path.join(model_path, SavedModelFile))

#Testing with trained model
else:
    savepoint = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        savepoint.restore(sess, os.path.join(model_path, SavedModelFile))
        test_dt = []
        test_sentence= random.choice(Train_Sentences)
        test_st =processSentence(test_sentence)
        print("*********************************************************")
        print("Total Trained Instances => ", len(Train_Sentences))
        print("Sentence similarity with Euclidian distance=> ") 
        print("Test Sentense => ",test_sentence)
        print("*********************************************************")

        test_dt.append(test_st)
        test_res = sess.run(encoder_output, feed_dict={X: test_dt})
        train_res = sess.run(encoder_output, feed_dict={X: Train_Data_Vector})
        Test_set = test_res.tolist()
        Test_set = Test_set[0]

    def euclid_distance(similarity, X1 , X2):
        euclid_distance = []
        
        for r in range(len(X1)):
            sum = 0
            for i in range(len(X2)):
                sum += (X2[i]- X1[r][i])**2
    
            euclid_distance.append([np.sqrt(sum), Train_Sentences[r]])
        euclid_distance = sorted(euclid_distance)
        
        for i in range(similarity):
             print("Sentense:- ",i+1, "=> " , euclid_distance[i][1])
         
 
    
    euclid_distance(5,train_res,Test_set)         

