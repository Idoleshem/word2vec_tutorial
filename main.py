import torch 
from torch import nn
import torch.optim as optim

# import function 
from utils import (
    collect_context_target_pairs,
    preprocess_raw_text_file,
    vector_representation,
    visualize_words_embedding,
)

# --------------- settings hyperparameters ------------------------

# context-target pairs selection
context_window_size = 3
# minimum words appreance threshold = 2 (currently not implmented)

# Training hyperparameters
epochs = 50 # amount of iterations over the entire text (corpus)
learning_rate = 0.01
batch_size = 16
embedding_size = 300

# ----------- Data Preprocessing & Tokenization ----------------
# text path
text_file_path = 'content\pickle_rick_transcript.txt'
tokens = preprocess_raw_text_file(text_file_path)

# ----------- Generate Context-Target pairs for training -------------
context_target_pairs = collect_context_target_pairs(tokens, context_window_size)


# ----------- Arranging Vectors Represention ----------
vocabulary, word2index, index2word = vector_representation(tokens)

# ----------- Prepare data for training  ---------

X_train = [] # list for input (target) vectors
y_train = [] # list for output (context) vectors

for target, context in context_target_pairs:
    X_train.append(word2index[target])
    y_train.append(word2index[context])

# Convert to PyTorch tensors for the model
X_train = torch.LongTensor(X_train)
y_train = torch.LongTensor(y_train)

# ----------- Define the Skip-Gram Model architecture ---------
class Skip_Gram_Model(nn.Module):

    def __init__(self, vocabulary_size, embedding_size):
        super(Skip_Gram_Model, self).__init__()
        self.embeddings = nn.Embedding(vocabulary_size, embedding_size) # an embedding layer which responsible for map each word to a dense vector
        self.linear = nn.Linear(embedding_size, vocabulary_size) # a linear layer which responsible for taking those embeddings and predict the context word.

    def forward(self, context_word):
        output = self.embeddings(context_word)
        output = self.linear(output)

        return output
    

# ----------- Word Embeddings Visualization ---------
# init the model instance
model = Skip_Gram_Model(len(vocabulary), embedding_size=embedding_size)

# init the loss and optimizer functions
loss_function = nn.CrossEntropyLoss() # calculates the error rate between the predicted value and the original value
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # a stochastic gradient descent

# visualize the word embeddings before training
visualize_words_embedding(model,"Before Training",word2index)

# start the training process
for epoch in range(epochs):

    total_loss = 0 # restart loss to 0

    # iterate over batch size
    for i in range(0,len(X_train),batch_size):
        x = X_train[i:i+batch_size]
        y_true = y_train[i:i+batch_size]
        optimizer.zero_grad() # clear the gradients
        y_pred = model(x) # backpropagation in order to compute the gradients
        loss = loss_function(y_pred, y_true.view(-1))
        loss.backward()
        optimizer.step() # update model parameters
        total_loss += loss.item()

    print(f'Epoch num: {epoch+1}, loss value: {total_loss:.3f}')

    visualize_words_embedding(model,epoch,word2index)


    # init the model instance
model = Skip_Gram_Model(len(vocabulary), embedding_size=embedding_size)

# init the loss and optimizer functions
loss_function = nn.CrossEntropyLoss() # calculates the error rate between the predicted value and the original value
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # a stochastic gradient descent

