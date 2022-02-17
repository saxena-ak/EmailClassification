# EmailClassification
Spam and Non Spam Classification of SpamAssassin database

Email classification that distinguishes between Spam and Non-Spam emails is a great use case of NLP and ML Classification. To classify emails into Spam and Non-Spam, they are transformed(pre-processed) using various NLP techniques into usable format and then ML algorithms is applied on them for classification. 

## Project Description: 
In this project, I have trained Naive Bayes model, SVM and Random Forest models that will classify Spam and Non-Spam(Ham) emails on a training dataset. These models are then evaluated on a test dataset. We also test the trained model on example messages that presents interesting outcome.

## Data:
The data is gathered from https://spamassassin.apache.org/old/publiccorpus/   
This is an open source project by Apache called SpamAssassin. I have downloaded the latest version(date) of the data. There are 4 folders with email message files. Two are for spam mails and remaining two are for normal emails. 

Each file looks like emails and contains a lot of information such as <b>Email Header</b> containing information regarding who sent the email, to whom, the IP address etc. Later, it contains <b>Email Body</b>. The Spam and Non-Spam emails are provided in separate folders.

1. Non Spam (Ham) - 3901 
2. Spam - 1898

