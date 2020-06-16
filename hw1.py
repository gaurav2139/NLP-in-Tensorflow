#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk #imporing NLTK library
nltk.download('gutenberg')


# ## Gutenberg corpus
# 
# NLTK includes a small selection of texts from the Project Gutenberg electronic text archive, which contains some 25,000 free electronic books, hosted at http://www.gutenberg.org/. We begin by getting the Python interpreter to load the NLTK package, then ask to see nltk.corpus.gutenberg.fileds()

# In[2]:


aic = gutenberg.words("lewis-carroll-alice-in-wonderland.txt") #if gutenberg not defined this erros shows up


# In[3]:


from nltk.corpus import gutenberg


# In[8]:


gutenberg.fileids() #available books in the gunteberg project


# Let's display other information about each text, by looping over all the values of fileid corresponding to the gutenberg file identifiers listed earlier and then computing statistics for each text. For a compact output display, we will round each number to the nearest integer, using round().

# In[13]:


for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid)) 
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    #no of character per word, no of word per sentecnes, no of words per vocabulary and the fields will be displayed
    print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)


# In[18]:


aic = gutenberg.words('carroll-alice.txt') #alice in wonderland gutenberg project
print("The length of Alice in Wonderland:",len(aic))
aic_sentences = gutenberg.sents('carroll-alice.txt')
print("The sentences present in the books are:",len(aic_sentences))
print("Paragraphs count:",len(gutenberg.paras('carroll-alice.txt')))


# In[19]:


from nltk.tokenize import word_tokenize #importing word tokenizer


# In[21]:


word_tokenize('carroll-alice.txt')


# In[23]:


from nltk.text import Text  #import text from nltk


# In[24]:


corpus = Text(gutenberg.words('carroll-alice.txt')) #importing dataset alice in wonderland from gutenberg project


# In[29]:


gutenberg.words('carroll-alice.txt')


# In[28]:


gutenberg.sents('carroll-alice.txt')    #senteces in alice in wonderland


# In[22]:


from nltk.stem import PorterStemmer
ps = PorterStemmer()


# In[30]:


dir(ps)   #directories in porter stemmer


# In[35]:


#example for stemming
print(ps.stem('stemming'))
print(ps.stem('walking'))
print(ps.stem('climbing'))
print(ps.stem('crying'))  #doesn't work for all the cases
print(ps.stem('sleeping'))


# In[36]:


nltk.download('averaged_perceptron_tagger') #shows error if you don't download the library


# In[37]:


#root word and tagging using Porter Stemmer
for w in aic:
    rootword = ps.stem(w)
    tagged = nltk.pos_tag(rootword)
    print(tagged)


# In[38]:


from nltk.stem import LancasterStemmer
stmLC = LancasterStemmer()


# In[39]:


print(stmLC.stem('stemming'))
print(stmLC.stem('walking'))
print(stmLC.stem('climbing'))
print(stmLC.stem('crying'))  #works compatitivley well than porter stemmer
print(stmLC.stem('sleeping'))


# In[40]:


#root word and tagging using lancaster Stemmer 
for w in aic:
    rootword = stmLC.stem(w)
    tagged = nltk.pos_tag(rootword)
    print(tagged)


# In[42]:


from nltk.stem import RegexpStemmer
st = RegexpStemmer('ing$|s$|e$|able$', min=4)


# In[43]:


print(st.stem('stemming'))
print(st.stem('walking'))
print(st.stem('climbing'))
print(st.stem('crying')) 
print(st.stem('sleeping'))


# In[44]:


#root word and tagging using Regular Expression stemmer
for w in aic:
    rootword = st.stem(w)
    tagged = nltk.pos_tag(rootword)
    print(tagged)


# In[46]:


nltk.download('rslp') #shows error if not downloaded


# In[47]:


from nltk.stem import RSLPStemmer
rslp = RSLPStemmer()


# In[50]:


nltk.download('indian') #indian corpus


# In[51]:


nltk.corpus.indian.words('hindi.pos')


# ### Annotated Text Corpora
# 
# Many text corpora contain linguistic annotations, representing POS tags, named entities, syntactic structures, semantic roles, and so forth. NLTK provides convenient ways to access several of these corpora, and has data packages containing corpora and corpus samples, freely downloadable for use in teaching and research. 1.2 lists some of the corpora. For information about downloading them, see http://nltk.org/data. For more examples of how to access NLTK corpora, please consult the Corpus HOWTO at http://nltk.org/howto.
