#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Download the NLTK CMU dataset 


# In[4]:


nltk.download('cmudict')


# In[2]:


import nltk
entries = nltk.corpus.cmudict.entries()
len(entries) #number of entries


# In[3]:


entries[:50] #here are the last fifty entries


# In[5]:


nltk.download('wordnet')


# In[6]:


from nltk.corpus import wordnet as wn
wn.synsets('abandon')


# In[7]:


wn.synset('abandon.n.01').lemma_names()


# In[8]:


wn.synset('abandon.v.01').lemma_names()


# In[9]:


wn.synset('abandon.v.02').lemma_names()


# In[10]:


def gender_features(word):
    return {'last_letter':word[-1]}


# In[11]:


gender_features('obama')


# In[12]:


nltk.download('names')


# ### Name Classifier

# In[54]:


from nltk.corpus import names
labeled_names = ([(name, 'male') for name in names.words('male.txt')]+
                 [(name, 'female') for name in names.words('female.txt')])


# In[14]:


import random
random.shuffle(labeled_names)


# In[15]:


featuresets = [(gender_features(n), gender) for (n,gender) in labeled_names]


# In[16]:


train_set, test_set = featuresets[500:], featuresets[:500]


# In[17]:


import nltk
classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.classify(gender_features('Louis'))


# In[52]:


a = classifier.classify(gender_features('obama'))
b = classifier.classify(gender_features('gaurav'))
c = classifier.classify(gender_features('trump'))
d = classifier.classify(gender_features('Bini'))
e = classifier.classify(gender_features('Meni'))
f = classifier.classify(gender_features('sonal'))
g = classifier.classify(gender_features('Avery'))


# #### Critical observation
# 
# * The model is classified in such a manner that the names ending with vowels always are considered as female but the model fails when the name ending with vowel is a male, also if the female names don't end with a vowel all the common names are taken and classified in their respective category.
# 
# 
# * For example, Avery for a female is a common name and is classified into feamle name but we should also note that the name does not end with a vowel. So, the conclusion can be taken into the account that common names are classified into their respective category.
# 
# 
# * Look for the name Sonal, where it is classified into male although it's a female name, so one more observation can be made that the rule is followed for the names which is common in English speaking countries and may or may not work for countries whose native language is english so the accuracy is around 75%, still the model works fine most of the time  

# In[53]:


print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)


# In[55]:


print(nltk.classify.accuracy(classifier, test_set))  #accuracy to test whether male or female


# In[32]:


from sklearn.feature_extraction.text import CountVectorizer   
#importing count vectorizer and transfroming into an numerical format


# In[57]:


vect = CountVectorizer(binary=True)
corpus = ["Teseract is good optical character recognition engine","optical character engine is significant"]
vect.fit(corpus)


# In[34]:


vocab = vect.vocabulary_


# In[35]:


#forming a key-value pair for the vocabulary
for key in sorted(vocab.keys()):
    print("{}:{}".format(key,vocab[key]))


# In[36]:


print(vect.transform(["This is a good optical illusion"]).toarray()) #tranform into vector format example 1


# In[37]:


print(vect.transform(corpus).toarray())  #tranform into vector format example 2


# In[38]:


from sklearn.metrics.pairwise import cosine_similarity  #import cosine similarity 


# In[58]:


similarity = cosine_similarity(vect.transform(["Google Cloud Vision is a character recognition engine"]).toarray(), vect.transform(["OCR is an optical character recognition engine"]).toarray())
similarity1 = cosine_similarity(vect.transform(["We develop state of art technology and transfrom today's world"]).toarray(), vect.transform(["We continue to serve our clients with state-of-the-art technology and a creative vision for tomorrow's challenges."]).toarray())


# In[59]:


print(similarity)   #cosine similarity for first one 
print(similarity1)  #cosine similarity for second one 


# In[43]:


similarity


# In[60]:


similarity1


# ### Check for the result
# 
# Comparing two local documents using cosine similarity, to check the result of cosine similarity we've used the same documents whether it gives the result zero or not

# In[61]:


similarity2 = cosine_similarity(vect.transform(["F:\M.Tech Sem 1-2\CSE6024 -- Machine Learning Techniques\DA\ML-DA.docx"]).toarray(), vect.transform(["F:\M.Tech Sem 1-2\CSE6024 -- Machine Learning Techniques\DA\ML-DA.docx"]).toarray())


# In[62]:


print(similarity2)
similarity2

