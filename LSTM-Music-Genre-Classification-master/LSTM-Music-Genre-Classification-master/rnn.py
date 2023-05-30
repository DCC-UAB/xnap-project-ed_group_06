#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    PyTorch implementation of a simple 2-layer-deep LSTM for genre classification of musical audio.
    Feeding the LSTM stack are spectral {centroid, contrast}, chromagram & MFCC features (33 total values)

    Question: Why is there a PyTorch implementation, when we already have Keras/Tensorflow?
    Answer:   So that we can learn more PyTorch and experiment with modulations on basic
              architectures within the space of an "easy problem". For example, SRU or SincNets.
              I'm am also curious about the relative performances of both toolkits.

"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from GenreFeatureData import (
    GenreFeatureData,
)  # local python class with Audio feature extraction (librosa)

# class definition
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=8, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # setup LSTM layers
        self.lstm = nn.RNN(self.input_dim, self.hidden_dim, self.num_layers, dropout = 0.5, bias = True) #dropout = 0.5

        # ---------------------batchnormalisation---------------------------------------
        self.batch = nn.BatchNorm1d(num_features = self.hidden_dim)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input, h):
        # lstm step => then ONLY take the sequence's final timetep to pass into the linear/dense layer
        # Note: lstm_out contains outputs for every step of the sequence we are looping over (for BPTT)
        # but we just need the output of the last step of the sequence, aka lstm_out[-1]
        lstm_out, hidden = self.lstm(input, h)
        logits = self.linear(lstm_out[-1])              # equivalent to return_sequences=False from Keras
        genre_scores = F.log_softmax(logits, dim=1)
        return genre_scores, hidden
    
    #--------------------------------------------------------------------------------------------
    
    def init_hidden(self, batch_size):
        " Initialize the hidden state of the RNN to zeros"
        return nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_dim))
    
    
    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
                torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()


def main():
    genre_features = GenreFeatureData()

    # if all of the preprocessed files do not exist, regenerate them all for self-consistency
    if (
            os.path.isfile(genre_features.train_X_preprocessed_data)
            and os.path.isfile(genre_features.train_Y_preprocessed_data)
            and os.path.isfile(genre_features.dev_X_preprocessed_data)
            and os.path.isfile(genre_features.dev_Y_preprocessed_data)
            and os.path.isfile(genre_features.test_X_preprocessed_data)
            and os.path.isfile(genre_features.test_Y_preprocessed_data)
    ):
        print("Preprocessed files exist, deserializing npy files")
        genre_features.load_deserialize_data()
    else:
        print("Preprocessing raw audio files")
        genre_features.load_preprocess_data()

    train_X = torch.from_numpy(genre_features.train_X).type(torch.Tensor)
    dev_X = torch.from_numpy(genre_features.dev_X).type(torch.Tensor)
    test_X = torch.from_numpy(genre_features.test_X).type(torch.Tensor)

    # Targets is a long tensor of size (N,) which tells the true class of the sample.
    train_Y = torch.from_numpy(genre_features.train_Y).type(torch.LongTensor)
    dev_Y = torch.from_numpy(genre_features.dev_Y).type(torch.LongTensor)
    test_Y = torch.from_numpy(genre_features.test_Y).type(torch.LongTensor)

    # Convert {training, test} torch.Tensors
    print("Training X shape: " + str(genre_features.train_X.shape))
    print("Training Y shape: " + str(genre_features.train_Y.shape))
    print("Validation X shape: " + str(genre_features.dev_X.shape))
    print("Validation Y shape: " + str(genre_features.dev_Y.shape))
    print("Test X shape: " + str(genre_features.test_X.shape))
    print("Test Y shape: " + str(genre_features.test_Y.shape))

    batch_size = 35  # num of training examples per minibatch
    num_epochs = 401

    # Define model
    print("Build Rnn RNN model ...")
    model = LSTM(
        input_dim=33, hidden_dim=128, batch_size=batch_size, output_dim=8, num_layers=2
    )
    
    #------------------------------------------------------------------------------
    loss_function = nn.NLLLoss()     #nn.NLLLoss()  # expects ouputs from LogSoftmax #nn.CrossEntropyLoss()

    #----------------------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.01) #weight_decay = 0.1
    #defineix com decau el lr
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, cycle_momentum= False)

    # To keep LSTM stateful between batches, you can set stateful = True, which is not suggested for training
    stateful = False

    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print("\nTraining on GPU")
    else:
        print("\nNo GPU, training on CPU")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # all training data (epoch) / batch_size == num_batches (12)
    num_batches = int(train_X.shape[0] / batch_size)
    num_dev_batches = int(dev_X.shape[0] / batch_size)

    val_loss_list, val_accuracy_list, epoch_list = [], [], []
    train_loss_list, train_accuracy_list = [], []

    print("Training ...")
        
    
    
    
    
    

    #Inicialització random i normalitzada
    # for name, w in model.named_parameters():
    #     if "weight_ih" in name:
    #         nn.init.xavier_uniform(w.data)
        
    #     if "bias" in name:
    #         nn.init.zeros_(w.data) 
    
    
    
    for name, w in model.named_parameters():
        if 'lstm' in name:
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(w.data)

            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(w.data)

            elif 'bias_ih' in name:
                nn.init.zeros_(w.data) 

            elif 'bias_hh' in name:
                nn.init.zeros_(w.data) 

        elif 'linear' in name:
            if 'weight' in name:
                nn.init.xavier_uniform_(w.data)

            elif 'bias' in name:
                nn.init.zeros_(w.data) 
    
    

    for epoch in range(num_epochs):

        train_running_loss, train_acc = 0.0, 0.0

        # Init hidden state - if you don't want a stateful LSTM (between epochs)
        
        #-----------------------------------------2 hidden layers--------------------------------
        #h_0, c_0 = model.init_hidden(batch_size)
        hidden_state = model.init_hidden(batch_size)


        for i in range(num_batches):
            #-------------------------------------------------
            #h_0, c_0 = h_0.to(device), c_0.to(device)
            hidden_state = hidden_state.to(device)
             
            
            # zero out gradient, so they don't accumulate btw batches
            model.zero_grad()

            # train_X shape: (total # of training examples, sequence_length, input_dim)
            # train_Y shape: (total # of training examples, # output classes)
            #
            # Slice out local minibatches & labels => Note that we *permute* the local minibatch to
            # match the PyTorch expected input tensor format of (sequence_length, batch size, input_dim)
            X_local_minibatch, y_local_minibatch = (
                train_X[i * batch_size: (i + 1) * batch_size, ],
                train_Y[i * batch_size: (i + 1) * batch_size, ],
            )
            # Reshape input & targets to "match" what the loss_function wants
            X_local_minibatch = X_local_minibatch.permute(1, 0, 2)

            # NLLLoss does not expect a one-hot encoded vector as the target, but class indices
            y_local_minibatch = torch.max(y_local_minibatch, 1)[1]

            X_local_minibatch, y_local_minibatch = X_local_minibatch.to(device), y_local_minibatch.to(device)
            
            #----------------------------------------------------------------------
            #y_pred, h_0, c_0 = model(X_local_minibatch, h_0, c_0)  # forward pass
            y_pred, hidden_state = model(X_local_minibatch, hidden_state)  # forward pass


            # Stateful = False for training. Do we go Stateful = True during inference/prediction time?
            #----------------------------------------------------------------
            #h_0.detach_(), c_0.detach_()
            
            hidden_state.detach_()
            

            loss = loss_function(y_pred, y_local_minibatch)  # compute loss
            loss.backward()  # backward pass
            optimizer.step()  # parameter update
            #scheduler.step()

            train_running_loss += loss.detach().item()  # unpacks the tensor into a scalar value
            train_acc += model.get_accuracy(y_pred, y_local_minibatch)

        print(
            "Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f"
            % (epoch, train_running_loss / num_batches, train_acc / num_batches)
        )

        if epoch % 10 == 0:
            print("Validation ...")  # should this be done every N=10 epochs
            val_running_loss, val_acc = 0.0, 0.0

            # Compute validation loss, accuracy. Use torch.no_grad() & model.eval()
            with torch.no_grad():
                model.eval()
                #------------------------------------------------
                #h_0, c_0 = model.init_hidden(batch_size)     
                hidden_state =  model.init_hidden(batch_size)

                for i in range(num_dev_batches):
                    #------------------------------------------
                    #h_0, c_0 = h_0.to(device), c_0.to(device)
                    hidden_state = hidden_state.to(device)


                    X_local_validation_minibatch, y_local_validation_minibatch = (
                        dev_X[i * batch_size: (i + 1) * batch_size, ],
                        dev_Y[i * batch_size: (i + 1) * batch_size, ],
                    )

                    X_local_minibatch = X_local_validation_minibatch.permute(1, 0, 2)
                    y_local_minibatch = torch.max(y_local_validation_minibatch, 1)[1]

                    X_local_minibatch, y_local_minibatch = X_local_minibatch.to(device), y_local_minibatch.to(device)

                    #---------------------------------------
                    #y_pred, h_0, c_0 = model(X_local_minibatch, h_0, c_0)
                    y_pred, hidden_state = model(X_local_minibatch, hidden_state)

                    

                    val_loss = loss_function(y_pred, y_local_minibatch)

                    val_running_loss += (
                        val_loss.detach().item()
                    )  # unpacks the tensor into a scalar value
                    val_acc += model.get_accuracy(y_pred, y_local_minibatch)

                model.train()  # reset to train mode after iterationg through validation data
                print(
                    "Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f | Val Loss %.4f  | Val Accuracy: %.2f"
                    % (
                        epoch,
                        train_running_loss / num_batches,
                        train_acc / num_batches,
                        val_running_loss / num_dev_batches,
                        val_acc / num_dev_batches,
                    )
                )

            epoch_list.append(epoch)
            val_accuracy_list.append(val_acc / num_dev_batches)
            val_loss_list.append(val_running_loss / num_dev_batches)
            train_accuracy_list.append(train_acc / num_batches)
            train_loss_list.append(train_running_loss / num_dev_batches)
     
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    @torch.no_grad()
    def evaluate(model, dev_X, dev_Y):
        prediccions = []
        y = []

        model.eval()

        h_0  = model.init_hidden(batch_size)                
        
        for i in range(num_dev_batches):

            h_0 = h_0.to(device)

            X_local_validation_minibatch, y_local_validation_minibatch = (
                dev_X[i * batch_size: (i + 1) * batch_size, ],
                dev_Y[i * batch_size: (i + 1) * batch_size, ],
            )

            X_local_minibatch = X_local_validation_minibatch.permute(1, 0, 2)
            y_local_minibatch = torch.max(y_local_validation_minibatch, 1)[1]

            X_local_minibatch, y_local_minibatch = X_local_minibatch.to(device), y_local_minibatch.to(device)

            y_pred, h_0 = model(X_local_minibatch, h_0)
                            
            pred = y_pred.data.max(1, keepdim=True)[1].cpu().numpy().tolist()
            prediccions += pred
            
            y += y_local_minibatch.cpu().numpy().tolist()

        return prediccions, y 
     
        

    prediccions, y = evaluate(model, dev_X, dev_Y)

    cm = confusion_matrix(y, prediccions)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [
        "classical",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
    ])
    disp.plot(xticks_rotation="vertical")
    plt.savefig("ConfPlotRNN.png")
    plt.show()
    plt.clf()


    # visualization loss
    plt.plot(epoch_list, val_loss_list, color = "red", label = "Val loss")
    plt.plot(epoch_list, train_loss_list, color = "blue", label = "Train loss")
    plt.xlabel("# of epochs")
    plt.ylabel("Loss")
    plt.title("RNN: Loss vs # epochs")
    plt.legend()
    plt.savefig('graphLossRNN.png')
    plt.show()
    plt.clf()

    # visualization accuracy
    plt.plot(epoch_list, val_accuracy_list, color="red", label = "Val Acc")
    plt.plot(epoch_list, train_accuracy_list, color = "blue", label = "Train Acc")
    plt.xlabel("# of epochs")
    plt.ylabel("Accuracy")
    plt.title("RNN: Accuracy vs # epochs")
    plt.legend()
    plt.savefig('graphAccuracyRNN.png')
    plt.show()


if __name__ == "__main__":
    main()
