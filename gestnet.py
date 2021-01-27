import glob
import matplotlib.pyplot as plt
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import BucketIterator, Field, TabularDataset
from tqdm import tqdm
import warnings


###########
# DATASET #
###########

class Dataset:  # Dataset object class utilizing Torchtext

    def __init__(self, path, batch_size=1):

        self.batch_size = batch_size
        self.input_field, self.output_field, self.data, self.data_iter = self.process_data(path)
        self.word2trialnum = self.make_trial_lookup(path)
        self.seg2ind = self.input_field.vocab.stoi  # from segment to torchtext vocab index

    def process_data(self, path):  # create Dataset object from tab-delimited text file

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        make_list = lambda b: [item for item in b.split(',')]
        make_float_list = lambda b: [float(item) for item in b.split(',')]


        input_field = Field(sequential=True, use_vocab=True, tokenize=make_list)  # morpheme segment format
        output_field = Field(sequential=True, use_vocab=False, tokenize=make_float_list)  # lip trajectory outputs

        datafields = [('underlying', None), ('surface', None), ('root_indices', None), ('suffix_indices', None),
                      ('word_indices', input_field), ('lip_output', output_field), ('tb_output', output_field)]

        data = TabularDataset(path=path, format='tsv', skip_header=True, fields=datafields)

        input_field.build_vocab(data, min_freq=1)
        data_iter = BucketIterator(data,
                                   batch_size=self.batch_size,
                                   sort_within_batch=False,
                                   repeat=False,
                                   device=device)

        return input_field, output_field, data, data_iter

    def make_trial_lookup(self, path):  # create lookup dictionary for use by make_trial method
        with open(path, 'r') as file:
            word2trialnum = {}
            for x, line in enumerate(file, start=-1):
                word2trialnum[line.split()[0]] = x
        return word2trialnum

    def make_trial(self, word):  # get target outputs for individual word for use by Seq2Seq's evaluate_word method

        trialnum = self.word2trialnum[word]

        source = self.data.examples[trialnum].word_indices
        lip_target = self.data.examples[trialnum].lip_output
        tb_target = self.data.examples[trialnum].tb_output

        source_list = []

        for seg in source:
            source_list.append(self.seg2ind[seg])

        source_tensor = torch.tensor(source_list, dtype=torch.long).view(-1, 1)

        lip_target_tensor = torch.tensor(lip_target, dtype=torch.double).view(-1, 1)
        tb_target_tensor = torch.tensor(tb_target, dtype=torch.double).view(-1, 1)

        return source_tensor, lip_target_tensor, tb_target_tensor


###########
# ENCODER #
###########

class Encoder(nn.Module):

    def __init__(self,
                 vocab_size,  # size of vector representing each segment in vocabulary (created by Dataset)
                 seg_embed_size,  # size of segment embedding
                 hidden_size):  # size of encoder hidden layer
        super(Encoder, self).__init__()

        self.params = (vocab_size, seg_embed_size, hidden_size)

        self.embedding = nn.Embedding(vocab_size, seg_embed_size)  # embedding dictionary for each segment
        self.rnn = nn.RNN(seg_embed_size, hidden_size)  # RNN hidden layer

    def forward(self, input_seq):
        embedded_seq = self.embedding(input_seq)
        output_seq, last_hidden = self.rnn(embedded_seq)
        return output_seq, last_hidden, embedded_seq


#############################
# ENCODER-DECODER ATTENTION #
#############################

class EncoderDecoderAttn(nn.Module):  # attention mechanism between encoder and decoder hidden states

    def __init__(self,
                 encoder_size,  # size of encoder hidden layer
                 decoder_size,  # size of decoder hidden layer
                 attn_size):  # size of attention vector
        super(EncoderDecoderAttn, self).__init__()  # always call this

        self.params = (encoder_size, decoder_size, attn_size)

        self.linear = nn.Linear(encoder_size+decoder_size, attn_size)  # linear layer

    def forward(self, decoder_hidden, encoder_outputs):

        decoder_hidden = decoder_hidden.squeeze(0)
        input_seq_length = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, input_seq_length, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attn = torch.tanh(self.linear(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))

        attn_sum = torch.sum(attn, dim=2)

        attn_softmax = F.softmax(attn_sum, dim=1).unsqueeze(1)
        attended_encoder_outputs = torch.bmm(attn_softmax, encoder_outputs).squeeze(1)

        encoder_norms = encoder_outputs.norm(dim=2)
        attn_map = attn_softmax.squeeze(1) * encoder_norms

        return attended_encoder_outputs, attn_map


###########
# DECODER #
###########

class Decoder(nn.Module):

    def __init__(self,
                 hidden_size,  # size of hidden layer for both encoder and decoder
                 attn):  # encoder-decoder attention mechanism
        super(Decoder, self).__init__()

        self.params = hidden_size

        self.output_size = 2  # number of articulators (lip and tongue body)

        self.attn = attn  # encoder-decoder attention mechamism
        self.rnn = nn.RNN(self.output_size+self.attn.params[0], hidden_size)  # RNN hidden layer
        self.linear = nn.Linear(hidden_size, self.output_size)  # linear layer

    def forward(self, input_tok, hidden, encoder_outputs):
        input_tok = input_tok.float()

        attended, attn_map = self.attn(hidden, encoder_outputs)
        rnn_input = torch.cat((input_tok, attended), dim=1).unsqueeze(0)

        output, hidden = self.rnn(rnn_input, hidden)
        output = self.linear(output.squeeze(0))

        return output, hidden, attn_map


##############################
# SEQUENCE TO SEQUENCE MODEL #
##############################

class Seq2Seq(nn.Module):  # Combine encoder and decoder into sequence-to-sequence model
    def __init__(self,
                 training_data=None,  # training data (Dataset class object)
                 load='',  # path to file for loading previously trained model
                 seg_embed_size=32,  # size of segment embedding
                 hidden_size=32,  # size of hidden layer
                 attn_size=32,  # size of encoder-decoder attention vector
                 optimizer='adam',  # what type of optimizer (Adam or SGD)
                 learning_rate=.001):  # learning rate of the model
        super(Seq2Seq, self).__init__()

        # Seq2Seq Parameters

        self.init_input_tok = nn.Parameter(torch.rand(1, 2))  # initialize first decoder input (learnable)

        # Hyperparameters / Device Settings

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_function = nn.MSELoss(reduction='sum')

        # Load a trained model and its subcomponents

        if load:
            self.path = re.sub('_[0-9]+.pt', '_', load)
            checkpoint = torch.load(load)

            self.encoder = Encoder(vocab_size=checkpoint['encoder_params'][0],
                                   seg_embed_size=checkpoint['encoder_params'][1],
                                   hidden_size=checkpoint['encoder_params'][2])

            attn = EncoderDecoderAttn(encoder_size=checkpoint['attn_params'][0],
                                      decoder_size=checkpoint['attn_params'][1],
                                      attn_size=checkpoint['attn_params'][2])

            self.decoder = Decoder(hidden_size=checkpoint['decoder_params'],
                                   attn=attn)

            self.loss_list = checkpoint['loss_list']

            self.load_state_dict(checkpoint['seq2seq_state_dict'])

            if checkpoint['optimizer_type'] == 'SGD':
                self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
            elif checkpoint['optimizer_type'] == 'Adam':
                self.optimizer = optim.Adam(self.parameters())
            else:
                print('Optimizer not loaded! Try again.')
                return

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        else:  # Initialize a new model and its subcomponents
            if not training_data:
                print('Required input: training_data (Dataset class object). Try again!')
                return

            self.path = None

            self.encoder = Encoder(vocab_size=len(training_data.input_field.vocab),
                                   seg_embed_size=seg_embed_size,
                                   hidden_size=hidden_size)

            attn = EncoderDecoderAttn(encoder_size=hidden_size,
                                      decoder_size=hidden_size,
                                      attn_size=attn_size)

            self.decoder = Decoder(hidden_size=hidden_size,
                                   attn=attn)

            self.loss_list = []

            for name, param in self.named_parameters():
                nn.init.uniform_(param.data, -0.08, 0.08)

            if optimizer == 'adam':
                self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            elif optimizer == 'sgd':
                self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
            else:
                'No such optimizer! Try again.'
                return

    def forward(self, input_seq, target_seq):
        input_length = input_seq.shape[0]
        target_length = target_seq.shape[0]
        target_output_size = self.decoder.output_size

        output_seq = torch.zeros(target_length, target_output_size).to(self.device)
        attn_map_seq = torch.zeros(target_length, input_length).to(self.device)

        encoder_outputs, hidden, embeddings = self.encoder(input_seq)

        input_tok = self.init_input_tok

        for t in range(target_length):
            output_tok, hidden, attn_map = self.decoder(input_tok,
                                                        hidden,
                                                        encoder_outputs)

            output_seq[t] = output_tok
            attn_map_seq[t] = attn_map

            input_tok = output_tok

        return output_seq, attn_map_seq

    def train_model(self, training_data, n_epochs=1):  # Train the model on the provided dataset

        self.train()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for _ in tqdm(range(n_epochs)):

                for i, batch in enumerate(training_data.data_iter):
                    self.zero_grad()
                    source = batch.word_indices
                    target_la = batch.lip_output
                    target_tb = batch.tb_output
                    target = torch.cat((target_la, target_tb), 1)
                    predicted, enc_dec_attn_seq = self(source, target)

                    loss = self.loss_function(predicted.float(), target.float())
                    loss.backward()
                    self.optimizer.step()

                self.loss_list.append(self.evaluate_model(training_data, verbose=False))

    def evaluate_model(self, training_data, verbose=True):  # Evaluate the model's performance on the dataset
        self.eval()
        epoch_loss = 0

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with torch.no_grad():

                for i, batch in enumerate(training_data.data_iter):
                    source = batch.word_indices
                    target_la = batch.lip_output
                    target_tb = batch.tb_output
                    target = torch.cat((target_la, target_tb), 1)
                    predicted, _ = self(source, target)

                    loss = self.loss_function(predicted.float(), target.float())
                    epoch_loss += loss.item()

            average_loss = epoch_loss / len(training_data.data_iter)

            if verbose:
                print(f'Average loss per word this epoch:')

            return average_loss

    def plot_loss(self):  # Plot the model's average trial loss per epoch
        plt.plot(self.loss_list, '-')
        plt.title('Average Trial Loss Per Epoch')
        plt.ylabel('Sum of Squared Error')
        plt.xlabel('Epoch')

    def evaluate_word(self, training_data, word, show_target=True):  # Evaluate the model's performance on a single word
        self.eval()

        trial = training_data.make_trial(word)

        with torch.no_grad():
            source = trial[0]
            target_la = trial[1]
            target_tb = trial[2]
            target = torch.cat((target_la, target_tb), 1)

            predicted, enc_dec_attn_seq = self(source, target)
            print(f'Target output:\n{target}')
            print(f'Predicted output:\n{predicted}')
            print(f'Encoder Decoder Attention:\n{enc_dec_attn_seq}')
        predicted_la = predicted[:, 0]
        predicted_tb = predicted[:, 1]

        figure_outputs, (lip_plot, tb_plot) = plt.subplots(2)

        figure_outputs.suptitle('Predicted Tract Variable Trajectories')

        # Lip Aperture Trajectory Subplot

        lip_plot.plot(predicted_la, label='Predicted')
        if show_target:
            lip_plot.plot(target_la, label='Target')
        lip_plot.set_title('Lip Tract Variable')
        lip_plot.set_ylabel('Constriction Degree (Lip Aperture)')
        lip_plot.set_ylim(10, -5)

        lip_plot.legend()

        # Tongue Body (Height) Trajectory Subplot

        tb_plot.plot(predicted_tb)
        if show_target:
            tb_plot.plot(target_tb)
        tb_plot.set_title('Tongue Body Height Tract Variable')
        tb_plot.set_xlabel('Time')
        tb_plot.set_ylabel('Constriction Degree (Height)')
        tb_plot.set_ylim(20, -5)

        # Plot Encoder-Decoder Attention

        heatmap_attn, ax = plt.subplots()
        heatmap_attn.suptitle('Encoder-Decoder Attention')
        im = ax.imshow(enc_dec_attn_seq.permute(1, 0), cmap='gray')

        ax.set_xticks([x for x in range(enc_dec_attn_seq.shape[0])])
        ax.set_xticklabels([x+1 for x in range(enc_dec_attn_seq.shape[0])])
        ax.set_xlabel('Decoder Time Point')

        ax.set_yticks([x for x in range(len(re.sub('-', '', word)))])
        ax.set_yticklabels(list(re.sub('-', '', word)))
        ax.set_ylabel('Input')

        plt.show()

    def save(self):  # Save model to 'saved_models'

        save_dict = {'encoder_params': self.encoder.params,
                     'attn_params': self.decoder.attn.params,
                     'decoder_params': self.decoder.params,
                     'seq2seq_state_dict': self.state_dict(),
                     'optimizer_type': str(self.optimizer)[0:4].strip(),
                     'optimizer_state_dict': self.optimizer.state_dict(),
                     'loss_list': self.loss_list}

        if not os.path.isdir('saved_models'):
            os.mkdir('saved_models')

        if self.path is None:
            model_num = 1
            while glob.glob(os.path.join('saved_models', f'gestnet_{model_num}_*.pt')):
                model_num += 1

            self.path = os.path.join('saved_models', f'gestnet_{model_num}_')
        else:
            model_num = self.path.split('_')[-2]

        saveas = f'{self.path}{str(len(self.loss_list))}.pt'

        torch.save(save_dict, saveas)
        print(f'Model saved as gestnet_{model_num}_{str(len(self.loss_list))} in directory saved_models.')

    def count_params(self):  # Count trainable model parameters
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('The model has ' + str(params) + ' trainable parameters.')


# Load a premade Dataset object for stepwise height harmony

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    data_stepwise = Dataset('trainingdata_stepwise.txt')
