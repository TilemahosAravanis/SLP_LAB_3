#!/usr/bin/python
import os
import warnings
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from sklearn.metrics import f1_score, recall_score, accuracy_score
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import matplotlib.pyplot as plt


########################################################
# Configuration
########################################################

# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50
EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "MR"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train1, X_test, y_test1 = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train1, X_test, y_test1 = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
le = LabelEncoder()
print("First 10 labels before encoding from the training set are:")
print(y_train1[:10])
y_train = le.fit_transform(y_train1)  # EX1
y_test = le.transform(y_test1)  # EX1
print("First 10 labels after encoding from the training set are:")
print(y_train[:10])
n_classes = le.classes_.size   # EX1 - LabelEncoder.classes_.size

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)
for i in range(10):
    print("\ntraining sample {}:".format(i))
    print(train_set.data[i])

for i in range(5):
    print("training sample {}\n\ndataitem = {} \nlabel = {}\n\nreturn values:\n\nexample = {}\nlabel = {}\nlength = {}\n============\n".format(
        i, X_train[i], y_train1[i], train_set[i][0], train_set[i][1], train_set[i][2]))

# EX7 - Define our PyTorch-based DataLoader
train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
model = BaselineDNN(output_size=n_classes,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
if (n_classes == 2):
    criterion = torch.nn.BCEWithLogitsLoss()  # EX8
else:
    criterion = torch.nn.CrossEntropyLoss()
parameters = model.parameters()
optimizer = torch.optim.Adam(parameters)  # EX8

#############################################################################
# Training Pipeline
#############################################################################

total_train_losses = []
total_test_losses = []

def get_metrics_report(y, y_hat):
    # report metrics
    report = f'  accuracy: {accuracy_score(y, y_hat)}\n  recall: ' + \
        f'{recall_score(y, y_hat, average="macro")}\n  f1-score: {f1_score(y, y_hat,average="macro")}'
    return report

for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_loss = train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)

    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)
    total_train_losses.append(train_loss)
    total_test_losses.append(test_loss)
    print(f"\n===== EPOCH {epoch} ========")
    print(f'\nTraining set\n{get_metrics_report(y_train_gold, y_train_pred)}')
    print(f'\nTest set\n{get_metrics_report(y_test_gold, y_test_pred)}')
    

# Plot
plt.figure(1)
plt.plot(range(1,EPOCHS+1), total_train_losses)
plt.plot(range(1,EPOCHS+1), total_test_losses)
plt.title('Learning curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train loss', 'test loss'])
plt.show()




