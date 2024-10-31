

# !pip install datasets
# !gdown https://drive.google.com/file/d/1-mB6idLW5Jg4aE68jOj5NDcDxRNlMXpu/view?usp=sharing --fuzzy

import torch
torch.manual_seed(0)

"""In this assigment we will use the data set wikitext. This dataset is composed of textual content extracted from Wikipedia articles which hopefully will lead to good embeddings when input in CBOW."""

from datasets import load_dataset
from random import sample

wikitext = load_dataset("wikipedia", "20220301.simple")
trim_dataset= wikitext['train']['text'][:5000]
# trim_dataset= sample(wikitext['train']['text'],5000)

"""Let's have a look at a datapoint."""

trim_dataset[0]

"""As you can see there are a lot of numeric values, end-of-lines and the word "References" at the end. We saw in the previous assigment how preprocessing can significantly reduce the vocabulary; feel free to base the following code in your previous work."""

import nltk
nltk.download('stopwords')

import string
import re
from nltk.corpus import stopwords
def preprocess_data(data):
  """ Method to clean text from noise and standarize text across the different classes.
      The preprocessing includes converting to joining all datapoints, lowercase, removing punctuation, and removing stopwords.
  Arguments
  ---------
  text : List of String
     Text to clean
  Returns
  -------
  text : String
      Cleaned and joined text
  """

  stop_words = set(stopwords.words('english'))

  #join all text in one single string
  text = ' '.join(data)

  #make everything lower case
  text =text.lower()

  #remove \n characters
  text = text.replace('\n', ' ')

  #remove word "References"
  # text = text.replace('references', '')
  text = re.sub(r'\breferences\b', '', text)

  #remove any punctuation or special characters
  text = re.sub(r'[^\w\s]', ' ', text)

  #remove all numbers
  text = re.sub(r'\d+', ' ', text)

  #remove all stopwords (see imports to help you with this)
  text = ' '.join(word for word in text.split() if word not in stop_words)

  return text


text=preprocess_data(trim_dataset)

# test
print(len(text.split()))

# test
text_test = ["AAA \n \n \n.  references. preferences  ;:!  #$% 123 sdfsdfsdf"]
print(preprocess_data(text_test))


def vocab_frequency(text):
  """ Creates dictionary of frequencies based on a dataset.
  Arguments
  ---------
  dataset : list of tuples
      list of tuples of the form (label, text)
  Returns
  -------
  vocab_dict : dictonary
      Dictionary of words and their frequencies with the format {word: frequency}
  """
  vocab_dict = {}
  #TODO same as assignment 1
  for word in text.split():
    if word not in vocab_dict:
      vocab_dict[word] = 1
    else:
      vocab_dict[word] += 1
  return vocab_dict


vocabulary = vocab_frequency(text)

len(vocabulary)




import torch
import torch.nn as nn
def word_to_index(vocabulary):
  """ Method to create vocabulary to index mapping.
  Arguments
  ---------
  vocabulary : Dictionary
     Dictonary of format {word:frequency}
  Returns
  -------
  word_to_index : Dictionary
      Dictionary mapping words to index with format {word:index}
  """
  word_to_index = {}
  #Create key,value pair for out of vocabulary worlds
  #TODO

  word_to_index['OOV'] = 0

  for index, word in enumerate(vocabulary, start = 1):
    word_to_index[word] = index

  return word_to_index


word_to_index = word_to_index(vocabulary)


def generate_dataset(data, window_size,word_to_index):
  """ Method to generate training dataset for CBOW.
  Arguments
  ---------
  data : String
     Training dataset
  window_size : int
     Size of the context window
  word_to_index : Dictionary
     Dictionary mapping words to index with format {word:index}
  Returns
  -------
  surroundings : N x W Tensor
      Tensor with index of surrounding words, with N being the number of samples and W being the window size
  targets : Tensor
      Tensor with index of target word
  """
  surroundings= []
  targets = []
  data= data.split(" ")
  #TODO complete function

  #get surrounding words based on window size
  #get target word (middle word)
  for i in range(window_size,len(data)-window_size):
    surrounding = []
    for j in range(i - window_size, i + window_size + 1):
      if j != i:
        surrounding.append(word_to_index.get(data[j], word_to_index['OOV']))
    target= word_to_index.get(data[i], word_to_index['OOV'])

    #append to surrounding
    #append to targets

    surroundings.append(surrounding)
    targets.append(target)

  surroundings = torch.tensor(surroundings, dtype=torch.long)
  targets = torch.tensor(targets, dtype=torch.long)

  return surroundings, targets

t_surroundings, t_targets = generate_dataset(text,5,word_to_index)

# test
print(t_surroundings)
print(t_targets)
print(len(t_surroundings))
print(len(t_targets))

print(len(text.split())) # should be bigger in 2*window_size


class CBOW(nn.Module):
  def __init__(self, vocab_size, embed_dim=300):
    """ Class to define the CBOW model
    Attributes
    ---------
    device : device
      Device where the model will be trained (gpu preferably)
    vocab_size : int
      Size of the vocabulary
    embed_dim : int
      Size of the embedding layer
    hidden_dim : int
      Size of the hidden layer
    """
    super().__init__()
    #use this layer to get a vector from the the word index
    self.embedding = nn.Embedding(vocab_size, embed_dim)

    #first fully connected layer (bottleneck)
    self.linear = nn.Linear(embed_dim, vocab_size)

  def forward(self, x):
    #pass input through embedding layer
    #average and resize (size must be batch_size x embed_dim)
    #pass through linear layer
    emb = self.embedding(x)
    average = emb.mean(dim=1)
    out = self.linear(average)
    return out

    return out

from torch.utils.data import DataLoader
#creation of dataloader for training
train_dataloader=DataLoader(list(zip(t_surroundings,t_targets)),batch_size=64,shuffle=True) #Here please change batch size depending of your GPU capacities (if GPU runs out of memory lower batch_size)

import time
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #CAUTION: RUN THIS CODE WITH GPU, CPU WILL TAKE TOO LONG
model = CBOW(len(word_to_index)).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10

#BE PATIENT: This code can take up to 2 hours and 10 min for a batch size of 64 and 10 epochs
start_time = time.time()
for epoch in range(epochs):
    total_loss = 0
    i=0
    for surr, tar in tqdm(train_dataloader):
        #TODO: create code for training our model
        surr = surr.to(device)
        tar = tar.to(device)
        optimizer.zero_grad()
        output = model(surr)
        loss = loss_function(output, tar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    print(f"Epoch {epoch+1} loss: {total_loss/len(train_dataloader)}")
end_time = time.time()
print(f"Time taken: {round((end_time - start_time)/3600,2)} hours")


def get_embedding(word, model, word_to_index):
  """ Method to get the embedding vector for a given word.
  Arguments
  ---------
  word : String
     Word given
  model : NN.module
     CBOW model
  word_to_index : Dictionary
     Dictionary mapping words to index with format {word:index}
  Returns
  -------
  word_embedding : Tensor
      Embedding vector for the given word
    """
  index = word_to_index.get(word, word_to_index['OOV']) # get word index

  with torch.no_grad():
        embedding_weights = model.embedding.weight# Get the weights of the embedding layer
        embedding_weights.requires_grad = False
        word_embedding = embedding_weights[index]# Extract the embedding vector for the given word index

  return word_embedding

print(get_embedding("shot",model,word_to_index)) #test this code by getting embedding of the word "shot"

def cosine_similarity(v1,v2):
  """ Method to calculate cosine similarity between two vectors.
  Arguments
  ---------
  v1 : Tensor
     First vector
  v2 : Tensor
     Second vector
  Returns
  -------
  cosine_similarity : float
      Cosine similarity between v1 and v2
  """
  dot_product = torch.dot(v1, v2)
  norm_v1 = torch.norm(v1)
  norm_v2 = torch.norm(v2)
  cosine_similarity = dot_product / (norm_v1 * norm_v2)
  return cosine_similarity.item()

def get_k_nearest_words(k, word, vocabulary,model,word_to_index):
  """ Method to find the k nearest words of a given vector
  Arguments
  ---------
  k : int
     Number of nearest words to return
  word : torch.Tensor
     Embedding vector for the given word
  vocabulary : Dictionary
     Dictionary mapping words to frequency with format {word:frequency}
  model : NN.module
     CBOW model
  word_to_index : Dictionary
     Dictionary mapping words to index with format {word:index}
  Returns
  -------
  similar : List of Strings
      List of k nearest words to the given word
  """
  device = next(model.parameters()).device
  word = word.to(device)  # Move result vector to device

  similarity_scores= torch.zeros(len(vocabulary), device=device)
  #fill similarity scores matrix using the word and our cosine_similarity function
  for i, vocab_word in enumerate(vocabulary.keys()):
    embedding = model.embedding(torch.tensor(word_to_index[vocab_word]).to(device))
    similarity_scores[i] = cosine_similarity(word, embedding)


  k_first= torch.topk(similarity_scores,k)
  similar=[]
  for i in k_first.indices:
    similar.append(list(vocabulary.keys())[i])
  #TODO: create a function to get the k nearest words to a certain chosen word. TIP: use pytorch's topk


  return similar



import pandas as pd
import time

def test_analogy(model, word_to_index, analogy_file):
  """ Method to test accuracy of CBOW embeddings on analogy tasks.
  Arguments
  ---------
  mode : nn.Module
     CBOW model
  word_to_index : Dictionary
     Dictionary mapping words to index with format {word:index}
  analogy_file : String
     File containing analogy tasks
  Returns
  -------
  accuracy : float
      accuracy of the model on the analogy tasks
  """
  device = next(model.parameters()).device

  df = pd.read_csv(analogy_file)
  df = df[df.category=='capital-common-countries'] #using capital cities subset of test set
  correct = 0
  total = 0
  for index,row in df.iterrows():

      word_one = row['word_one'].lower() #extract word number 1
      word_two = row['word_two'].lower()   #extract word number 2
      word_three = row['word_three'].lower() #extract word number 3
      word_four = row['word_four'].lower() #extract word number 4

      #remember to standarize the words by using .lower

      #create exception in case word is not in vocabulary
      if word_one not in word_to_index or word_two not in word_to_index or word_three not in word_to_index or word_four not in word_to_index:
        continue

      #get embedding of all words
      embedding_word_one = model.embedding(torch.tensor(word_to_index[word_one]).to(device))
      embedding_word_two = model.embedding(torch.tensor(word_to_index[word_two]).to(device))
      embedding_word_three = model.embedding(torch.tensor(word_to_index[word_three]).to(device))

      # calculate embedding_word_2-embedding_word_1+embedding_word_3
      result = embedding_word_two - embedding_word_one + embedding_word_three

      #call the k_nearest function you created before (set k to 10)
      # Get the top 10 most similar words to the result vector
      prediction = get_k_nearest_words(10, result, vocabulary, model, word_to_index)


      #if word_four is in prediction, add one to correct
      if word_four in prediction:
            correct += 1

      total+=1
      print(f"Total = {total}, Correct = {correct} temp_accuracy = {correct/total}")


  if total != 0:
    accuracy=correct/total
  else:
    return 'No word was found in the embeddings '
  return accuracy

start_time = time.time()
print(test_analogy(model,word_to_index,'TestSet_sample.csv'))
end_time = time.time()
print(f"Time taken: {round((end_time - start_time)/3600,2)} hours")