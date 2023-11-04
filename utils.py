# imports relevant for tokenization
import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt') # punctuation words '(),[].:?'
nltk.download('stopwords') # irrelevant words such as “a” “an”

# imports relevant for skip-gram model
import torch
from torch import nn
import torch.optim as optim

# imports relevant for visualiztion
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns



def preprocess_raw_text_file(text_file_path):

  # load the text
  with open(text_file_path, 'r') as file:
    text = file.read().replace('\n', ' ')

  # lowercase the text
  text = text.lower()

  # remove punctuations like '(),[].:?'
  text = text.translate(str.maketrans("", "", string.punctuation))

  # Ttokenization
  tokens = word_tokenize(text)

  # remove stop words ('at','so','an','or')
  stop_words = set(stopwords.words('english'))

  # collect all words which are not stop words and not numbers, keep the order
  tokens_after_filtering = []

  for token in tokens:

    if token not in stop_words and token.isnumeric() == False:
      tokens_after_filtering.append(token)


  return tokens_after_filtering


# create context-target pairs

def collect_context_target_pairs(tokens,context_window_size):

  context_target_pairs = []
  for i in range(context_window_size, len(tokens) - context_window_size):

    # set target (center) word
    target = tokens[i]

    # extract sublist with context words (-3,-2,-1,target,1,2,3)
    context = tokens[i-context_window_size:i+context_window_size+1]

    # remove the target word from context
    context.remove(target)

    # iterate over words in window
    for word in context:
        context_target_pairs.append((target, word))

  return context_target_pairs


def vector_representation(tokens):

  # get the unique tokens
  vocabulary = sorted(set(tokens))

  # map word to it's corresponding index
  word2index = {word: index for index, word in enumerate(vocabulary)}

  # map index to it's corresponding word
  index2word = {index: word for index, word in enumerate(vocabulary)}

  return vocabulary, word2index, index2word


def visualize_words_embedding(model,epoch_number,word2index):

  # select a small group of words
  words_to_visualize = ['rick','ricks','mooorty', 'morty', 'mortys','pickle','antipickle','pickles','angry','great','well','car','beth','goldenfold','wong','dr','guns','animal','jaguar','kids','geez','die','summer','god','serum','family','therapy','fuck','mr','office']

  # extract the model weights
  word_vectors = model.embeddings.weight.data

  # get the word embeddings
  indices = [word2index[word] for word in words_to_visualize]
  word_vectors = model.embeddings.weight.data[indices]

  # fit a 2d t-SNE model to the embedding vectors
  tsne = TSNE(n_components=2,perplexity=20)
  word_vectors_2d = tsne.fit_transform(word_vectors)

  # get a specific color for each dot
  colors = sns.husl_palette(n_colors = len(words_to_visualize))

  # create a scatter plot of the projection
  plt.figure(figsize=(12,12))

  # annotate the points on the graph
  for i, word in enumerate(words_to_visualize):
    plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1],c=colors)
    plt.annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]))

  plt.title(f'Pickle Rick Word Embeddings Visualization - {epoch_number}')

  # Save the figure
  plt.savefig(f'word_embeddings_epoch_{epoch_number}.png')