# EmailClassification
Spam and Non Spam Classification of SpamAssassin database

Email classification that distinguishes between Spam and Non-Spam emails is a great use case of NLP and ML Classification. To classify emails into Spam and Non-Spam, the data are transformed(pre-processed) using various NLP techniques into usable format and trained using ML algorithms for classification such as Naive Bayes. 

## Project Description: 
In this project, I have trained 3 different Machine Learning models - 1) Naive Bayes model, 2) Support Vector Machine(SVM) and 3) Random Forest on the pre-processed data set. These models are then evaluated on seperate a test dataset and the results are presented. I have also tested the trained model on few example messages that presents interesting outcome.

## Data:
The data is gathered from https://spamassassin.apache.org/old/publiccorpus/   

This is an open source project by Apache called SpamAssassin. I have downloaded the latest version(date) of the data. There are 4 folders with email message files. Two are for spam mails and remaining two are for normal emails. 

Each file looks like emails and contains a lot of information such as <b>Email Header</b> containing information regarding who sent the email, to whom, the IP address etc. Later, it contains <b>Email Body</b>. The Spam and Non-Spam emails are provided in separate folders.

1. Non Spam (Ham) - 3901 
2. Spam - 1898

# How to use the code:

1. Create 'SpamData/01_Processing/spam_assassin_corpus/' directories. 
2. Download the latest dataset from https://spamassassin.apache.org/old/publiccorpus/ and store in the 'spam_assassin_corpus' folder.
3. Unzip/Extract the dataset in the same folder. 
4. Download and run the python code.
