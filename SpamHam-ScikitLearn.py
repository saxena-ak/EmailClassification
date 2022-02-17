#!/usr/bin/env python
# coding: utf-8

# # Spam Ham Email Classification using Naive Bayes, Random Forest and SVM Classifiers
# 
# Email classification that distinguishes between Spam and Non-Spam emails is a great use case of NLP and ML Classification. To classify emails into Spam and Non-Spam, they are transformed(pre-processed) using various NLP techniques into usable format and then ML algorithm is applied on them. 
# 
# ## Project Description: 
# In this project, I will be creating Naive Bayes model, SVM and Random Forest models that will classify Spam and Non-Spam(Ham) emails. 

# In[301]:


import numpy as np
import pandas as pd
from os import walk
from os.path import join
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score

import re

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


# In[134]:


Data_JSON_Path = 'SpamData/01_Processing/email-text-data 3.json'

#Email corpus file path are mentioned below
Spam_1_Path = 'SpamData/01_Processing/spam_assassin_corpus/spam_1'
Spam_2_Path = 'SpamData/01_Processing/spam_assassin_corpus/spam_2'
Easy_Ham1_Path = 'SpamData/01_Processing/spam_assassin_corpus/easy_ham_1'
Easy_Ham2_Path = 'SpamData/01_Processing/spam_assassin_corpus/easy_ham_2'

Data_JSON_Path = 'SpamData/01_Processing/email-text.json'

Spam_Class = 1
Ham_Class = 0


# ## Data:
# The data is gathered from https://spamassassin.apache.org/old/publiccorpus/   
# This is an open source project by Apache called SpamAssassin. I have downloaded the latest version(date) of the data.   
# 
# Let us explore the data a bit.

# In[135]:


#Opening a file. The path is saved in constant Sample_File.
stream = open("SpamData/01_Processing/practice_email.txt", encoding='utf-8')
#Here the encoding is utf-8 or unicode encoding. 
msg = stream.read()
stream.close()
print(msg)


# As we can see, the above data looks like an email. It contains a lot of information such as <b>Email Header</b> containing information regarding who sent the email, to whom, the IP address etc. Later, it contains <b>Email Body</b>. The Spam and Non-Spam emails are provided in separate folders.
# 

# Now, I will extract only text body from the emails and store them in a dataframe using the below function which is a Generator Function. Also, adding the Labels or classification - Spam and Ham.
# 
# The below function email_body_generator() yields the email body and file name from the path specified. The function - df_from_dir creates a dataframe of all the files, email body and classification.

# In[136]:


def email_body_generator(path):
    #Looping through all the files in the path
    for root, dirnames, filenames in walk(path):
        for file_name in filenames:
            #Finding the address
            filepath = join(root, file_name)
            #setting the encoding to latin-1
            stream = open(filepath, encoding='latin-1')
            is_body = False
            lines = []
            #Since the body starts after a space, using \n to identify it
            for line in stream:
                if is_body:
                    lines.append(line)
                elif line == '\n':
                    is_body = True

            email_body = '\n'.join(lines)
            #Generator function yields instead of return
            yield file_name, email_body


# In[137]:


def df_from_dir(path, classification):
    body = []
    name = []
    # Storing the Message and Category in a dataframe 
    # and setting the File Name as index of the DF
    for file_name, email_body in email_body_generator(path):
        body.append({'Message':email_body, 'Category': classification})
        name.append(file_name)
    return pd.DataFrame(body, index = name)

Since there are two Folders each for Spam and Ham, calling method and storing data in DF and appending files of second folder in first DF. 
# In[138]:



#Spam Folders
spam_emails = df_from_dir(Spam_1_Path,Spam_Class)
spam_emails = spam_emails.append(df_from_dir(Spam_2_Path,Spam_Class))

#Ham Folders
ham_emails = df_from_dir(Easy_Ham1_Path,Ham_Class)
ham_emails = ham_emails.append(df_from_dir(Easy_Ham2_Path,Ham_Class))


# In[139]:


data = pd.concat([spam_emails,ham_emails])
print("Spam mails",spam_emails.shape,"Ham emails", ham_emails.shape, "Total emails",data.shape)


# In[140]:


data.index.name = 'Doc_Name'
data


# # Data Cleaning
# 
# Checking for missing values or empty emails. This is an important step as we would like to avoid any such emails and remove them from our dataset. Thses kind of emails will not be helpful during training or testing phase. 

# In[141]:


# If message body = Null
data.Message.isnull().values.any()


# As we can see from the above that <b>No missing</b> values are found. 
# 
# We can also check for empty emails or emails that may have empty spaces. 
# 

# In[142]:


(data.Message.str.len() ==0).sum()


# There are 3 empty emails found. Let us go more into details and look for these empty emails. We can find out the message body of these emails to analyse further. 
# 
# The message body resulted into the below index where we can see that the email body is 'cmds'. If we go back to where all the emails are stored, we can see these are not email files but some system files that were generated during unzipping process. We can get rid of them now. 

# In[143]:


#Locating the empty emails
data[data.Message.str.len() ==0].index


# In[144]:


#remove system file entries from DF
print("Shape of data--> Before", data.shape)
data.drop(['cmds'], inplace =True)
print("Shape of data--> After", data.shape)


# It can be seen now that there are no empty emails left now.
# 
# # Adding document IDs instead of ID present in dataset
# 
# Also, moving the FileName to a new column called File_Name to make it a more nice to read and use dataframe.

# In[145]:


document_ids = range(0,len(data.index))
data['Doc_ID'] = document_ids
data['File_Name'] = data.index
data.set_index('Doc_ID',inplace=True)
data


# Now, this dataframe contains all the emails, file names, category or the labels and Document ID. We will store it in a JSON format so that the same data can be used later if requires. This also creates our first checkpoint into the code.
# 
# ## Checkpoint - 1

# In[146]:


#Storing the data in JSON format. It can be used later.
data.to_json(Data_JSON_Path)


# # Data Visualization
# 
# Visualizing the percentage of Spam and Non-Spam emails

# In[147]:


no_spam = data.Category.value_counts()[1]
no_ham = data.Category.value_counts()[0]

category_names = ['Spam', 'Non Spam']
sizes = [no_spam,no_ham]
cus_colors = ['#eb3b5a','#3867d6']
plt.figure(figsize = (5,5), dpi = 80)
plt.pie(sizes, labels = category_names, textprops={'fontsize':12}, startangle = 50, 
        autopct = '%1.2f%%', colors = cus_colors, explode = [0.01,0.01], pctdistance = 0.8)

#Created circle, supplied size, color and where should it be drawn
circle_centre = plt.Circle((0,0), radius = 0.6, fc = 'white')
#we get current axis, add a circle on top of current axis. 
plt.gca().add_artist(circle_centre)

plt.show()


# # Natural Language Processing(NLP)
# 
# Preprocessing the text using NLP is required as we can not use text data for ML model. We need to convert it into format that can be used in ML model. For that, we will be doing the below tasks.As we can see, the data has a lot of HTML tags, spaces, extra characters, so we will try to remove them in addition to the NLP.
# 
# In the following function we try to achieve the below tasks:
# 
# 1. Stripping HTML tags, Special Character, single characters, multiple spaces, etc.
# 2. Convert text into lower case
# 3. Stop word removal - I, the, is, me etc.
# 4. Stripping out HTML
# 5. Word Stemming - reduce the word to its stem word using WordNetStemmer
# 
# 
# 
# 7. Remove the punctuation
# 
# ### Download NLTK resources like Tokenizer and Stopwords to use them for preprocessing

# In[276]:


nltk.download('wordnet')
nltk.download('punkt')
stemmer = PorterStemmer()
def clean_text(text):
    documents = []
    stemmer = PorterStemmer()
    #stemmer = WordNetLemmatizer()
    #print("text is",text)
    for i in range(0,len(text)):
        
        soup = BeautifulSoup(text[i], 'html.parser')
        document = soup.get_text()
        
        # Remove all the special characters
        #document = re.sub(r'\W', ' ', str(text[i]))
        document = re.sub(r'\W+', ' ', str(document))
        
        document = re.sub(r"[^a-zA-Z0-9]+", ' ', document)

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()
        document = [stemmer.stem(word) for word in document]
        #document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)
    return documents


# In[277]:


get_ipython().run_cell_magic('time', '', '#Message = data.Message.apply(clean_text)\nMsg = clean_text(data.Message)')


# In[278]:


data.Message = Msg


# In[279]:


data


# In[280]:


data.sort_index(inplace = True)


# # Tokenising - Bag of Words approach 

# In[281]:


tfidfconverter = TfidfVectorizer(stop_words='english')
all_features = tfidfconverter.fit_transform(data.Message)


# In[282]:


all_features.shape


# In[283]:


len(tfidfconverter.vocabulary_)


# In[284]:


X_train, X_test,y_train,y_test = train_test_split(all_features,data.Category, test_size =0.3, random_state=88)


# In[285]:


X_train.shape


# #### Now that the test and train data is ready, we will train 3 models and compare and analyse the results. For the Evaluation, following matrices are used:
# 1. Accuracy
# 2. Precision
# 3. Recall
# 4. F1-Score

# # Model 1: Naive Bayes Classifier

# In[286]:


NB_classifier = MultinomialNB()
NB_classifier.fit(X_train,y_train)
y_predicted_NB = NB_classifier.predict(X_test)
#Calculating the Accuracy of the model
NB_classifier.score(X_test,y_test)
scores = precision_recall_fscore_support(y_test,y_predicted_NB)


# Recall and Precision for classifier

# In[324]:


print("Scores for class Spam")
print("Accuracy of Naive Bayes is:",round(NB_classifier.score(X_test,y_test),4))
print(f"Scores for Class 0 or Non-Spam: Precision: {round(scores[0][0],2)} Recall: {round(scores[1][0],2)} F1-Scores: {round(scores[2][0],2)}")
print(f"Scores for Class 1 or Spam: Precision: {round(scores[0][1],2)} Recall: {round(scores[1][1],2)} F1-Scores: {round(scores[2][1],2)}")


# # Model 2: Random Forest Classifier:

# In[325]:


classifier_RF = RandomForestClassifier(n_estimators=100, max_depth=None,n_jobs=-1)
classifier_RF.fit(X_train, y_train) 
y_predicted_RF = classifier_RF.predict(X_test)
scores_RF = precision_recall_fscore_support(y_test,y_predicted_RF)
print("Scores for class Spam")
print("Accuracy of Random Forest is:",round(classifier_RF.score(X_test,y_test),4))
print(f"Scores for Class 0 or Non-Spam: Precision: {round(scores_RF[0][0],2)} Recall: {round(scores_RF[1][0],2)} F1-Scores: {round(scores_RF[2][0],2)}")
print(f"Scores for Class 1 or Spam: Precision: {round(scores_RF[0][1],2)} Recall: {round(scores_RF[1][1],2)} F1-Scores: {round(scores_RF[2][1],2)}")


# # Model 3: Suppport Vector Machine

# In[327]:


clf = SVC(kernel='linear').fit(X_train, y_train)
y_predicted_SVM = clf.predict(X_test)
scores_SVM = precision_recall_fscore_support(y_test,y_predicted_SVM)
print("Scores for class Spam")
print("Accuracy of SVM is:",round(clf.score(X_test,y_test),4))
print(f"Scores for Class 0 or Non-Spam: Precision: {round(scores_SVM[0][0],2)} Recall: {round(scores_SVM[1][0],2)} F1-Scores: {round(scores_SVM[2][0],2)}")
print(f"Scores for Class 1 or Spam: Precision: {round(scores_SVM[0][1],2)} Recall: {round(scores_SVM[1][1],2)} F1-Scores: {round(scores_SVM[2][1],2)}")


# # Testing the model created to predict some spammy (& non-spammy) emails
# 
# Writing down some example emails in a list to test our 3 models:

# In[290]:


sample_email = ['Need house loan? get quotes on low interest rates. Call now',
                'Hi, Can you please help me with the task we discussed in the team meeting?',
                'Learn how to loose 15 kg in 1 week',
                'Hi, how is 5:00 pm for our lunch on Wednesday? I am pretty busy on other days throughout the week',
                'get viagra for free now']


# In[291]:


sample_email_clean = clean_text(sample_email)
sample_email_clean


# In[292]:


sample_emails = tfidfconverter.transform(sample_email_clean)


# Transforming the sample emails using tfidf

# In[294]:


sample_email_features = tfidfconverter.transform(sample_emails)


# ## Predicting example cases using Naive Bayes Classifier

# In[295]:


NB_classifier.predict(sample_email_features)


# The model predicts the email text 1 and 5 as Spams. The remaining are classified as non spams. However, upon further investigation, email 3 also looks spam. This could be due to out training data did not have enough samples to categorize the words in email 3 as spams

# ## Predicting example cases using Random Forest Classifier

# In[296]:


classifier_RF.predict(sample_email_features)


# In[297]:


classifier_RF.predict(sample_email_f)


# The model predicts all the emails as spams. This could be due to RF not suitable for a Sparse Matrix. For sparse data, it is possible that for a node, the bootstrapped sample and the random subset of features will collaborate to produce an invariant feature space. There's no productive split, so it's unlikely that the children of this node will be at all helpful. Even though the accuracy is higher than the Random Forest, its performance on example emails is not up to the mark. Another reason could be class imbalance as ~30% classes are spam and ~70% are non spam emails.

# ## Predicting using Support Vector Machine

# In[298]:


clf.predict(sample_email_features)


# SVM creates a line or a hyperplane which separates the data into classes. Each data is plotted as a point in n dimernsional space where n is number of features. SVM is successfully able to differntiate between two different classes. As we can see above, it classified all the spam and non spam example messages correctly. 
