# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:54:00 2020

@author: LFlueckiger
"""


import pandas as pd

# load labeled email data and get familiar with it
# source: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
sms_spam = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['Label', 'SMS'])
print(sms_spam.shape)
print(sms_spam.head())

# check percentage of spam and ham
print(sms_spam['Label'].value_counts(normalize=True))


#%% Split into training/testing
# Randomize the dataset
data_randomized = sms_spam.sample(frac=1, random_state=1)

# Calculate index for 80% training / 20% test split
training_test_index = round(len(data_randomized) * 0.8)

# split into Training/Test 
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)
print(training_set.shape)
print(test_set.shape)

# check percentage of spam and ham for training and test set
print(training_set['Label'].value_counts(normalize=True))
print(test_set['Label'].value_counts(normalize=True))


#%% Clean data
# removing all the punctuation and bringing every letter to lower case
training_set['SMS'] = training_set['SMS'].str.replace('\W', ' ')
training_set['SMS'] = training_set['SMS'].str.lower()

# split SMS into list of words
training_set['SMS'] = training_set['SMS'].str.split()

# list all the unique words in the training set
vocabulary = []
for sms in training_set['SMS']:
    for word in sms:
        vocabulary.append(word)
        
vocabulary = list(set(vocabulary))
print("Number of unique words in vocabulary list: ", len(vocabulary))

# create dictionary of zero unique words in each sms 
word_counts_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

# fill dictionary with word count
for index, sms in enumerate(training_set['SMS']):
    for word in sms:
        word_counts_per_sms[word][index] += 1

# convert dictionary to dataframe
word_counts = pd.DataFrame(word_counts_per_sms)

# combine word count set with training set
training_set_clean = pd.concat([training_set, word_counts], axis=1)
print("Final training set:\n", training_set_clean.head())


#%% Calculating conditional probability value associated with each word in the vocabulary
# Isolating spam and ham messages first
spam_messages = training_set_clean[training_set_clean['Label'] == 'spam']
ham_messages = training_set_clean[training_set_clean['Label'] == 'ham']

# calculate propability of Spam and Ham
p_spam = len(spam_messages) / len(training_set_clean)
p_ham = len(ham_messages) / len(training_set_clean)

# number of words in Spam, Ham and vocabulary
n_words_per_spam_message = spam_messages['SMS'].apply(len)
n_spam = n_words_per_spam_message.sum()

# N_Ham
n_words_per_ham_message = ham_messages['SMS'].apply(len)
n_ham = n_words_per_ham_message.sum()

# N_Vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing constant
alpha = 1

# Initiate dictionary of zero propability for each word
parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}

# Calculate count of unique words and resulting propability value for spam and ham
for word in vocabulary:
    n_word_given_spam = spam_messages[word].sum()   
    p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
    parameters_spam[word] = p_word_given_spam
    
    n_word_given_ham = ham_messages[word].sum()   
    p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
    parameters_ham[word] = p_word_given_ham
    
    
#%% Build spam classifier for new message   
import re

# define the classifier
def classify(message):
    # message input has to be string

    # remove punctuation, bringing every letter to lower case and split
    message = re.sub('\W', ' ', message)
    message = message.lower().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    # multiply probability values for each word in message
    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
            
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]
            
    print('P(Spam|message):', p_spam_given_message)
    print('P(Ham|message):', p_ham_given_message)
    
    # classify as spam or ham 
    if p_ham_given_message > p_spam_given_message:
        print('Label: Ham')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal proabilities, have a human classify this!')    

# test classifier on two messages                
classify('WINNER!! This is the secret code to unlock the money: C3421.')  

classify("Sounds good, Tom, then see u there")


#%% Build spam filter for test set
def classify_test_set(message): 
    # message input has to be string
    
    # remove punctuation, bringing every letter to lower case and split
    message = re.sub('\W', ' ', message)
    message = message.lower().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
            
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]
    
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_spam_given_message > p_ham_given_message:
        return 'spam'
    else:
        return 'needs human classification'


# Apply classifier to test set  
test_set['predicted'] = test_set['SMS'].apply(classify_test_set)
print(test_set.head())


#%% determine accuracy of spam filter
correct = 0
total = test_set.shape[0]

# check for each sms if classifier was correct    
for row in test_set.iterrows():
    row = row[1]
    if row['Label'] == row['predicted']:
        correct += 1
        
print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)




