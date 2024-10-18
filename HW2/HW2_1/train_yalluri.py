import torch.optim as optim
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from scipy.special import expit
import sys
import os
import json
import re
import pickle
from torch.utils.data import DataLoader, Dataset


def preprocess_data():
    filepath = 'data/'
    with open(filepath + 'training_label.json', 'r') as f:
        file = json.load(f)

    word_count = {}
    for d in file:
        for s in d['caption']:
            word_sentence = re.sub('[.!,;?]', ' ', s).split()
            for word in word_sentence:
                word = word.replace('.', '') if '.' in word else word
                word_count[word] = word_count.get(word, 0) + 1

    word_dict = {word: count for word, count in word_count.items() if count > 4}
    useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {i + len(useful_tokens): w for i, w in enumerate(word_dict)}
    w2i = {w: i + len(useful_tokens) for i, w in enumerate(word_dict)}
    
    for token, index in useful_tokens:
        i2w[index] = token
        w2i[token] = index
        
    return i2w, w2i, word_dict

def process_sentence(sentence, word_dict, w2i):
    sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
    sentence = [w2i.get(word, 3) for word in sentence]
    sentence.insert(0, 1)
    sentence.append(2)
    return sentence

def prepare_annotations(label_file, word_dict, w2i):
    label_json = 'data/' + label_file
    annotated_caption = []
    with open(label_json, 'r') as f:
        label = json.load(f)
    for d in label:
        for s in d['caption']:
            s = process_sentence(s, word_dict, w2i)
            annotated_caption.append((d['id'], s))
    return annotated_caption

def load_avi_data(files_dir):
    avi_data = {}
    training_feats = 'data/' + files_dir
    files = os.listdir(training_feats)
    for file in files:
        value = np.load(os.path.join(training_feats, file))
        avi_data[file.split('.npy')[0]] = value
    return avi_data

def batch_data(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths

class CustomTrainDataset(Dataset):
    def __init__(self, label_file, files_dir, word_dict, w2i):
        self.label_file = label_file
        self.files_dir = files_dir
        self.word_dict = word_dict
        self.avi = load_avi_data(label_file)
        self.w2i = w2i
        self.data_pair = prepare_annotations(files_dir, word_dict, w2i)
        
    def __len__(self):
        return len(self.data_pair)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000)/10000.
        return torch.Tensor(data), torch.Tensor(sentence)

class CustomTestDataset(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])
    def __len__(self):
        return len(self.avi)
    def __getitem__(self, idx):
        return self.avi[idx]

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)
        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        attention_weights = self.to_weight(x).view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.compress = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(512, 512, batch_first=True)

    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()    
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, 512)
        output, hidden_state = self.gru(input)
        return output, hidden_state

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage=0.3):
        super(Decoder, self).__init__()
        self.hidden_size = 512
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.embedding = nn.Embedding(output_size, 1024)
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(hidden_size+word_dim, hidden_size, batch_first=True)
        self.attention = AttentionMechanism(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long().cuda()
        seq_logProb, seq_predictions = [], []
        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()

        for i in range(seq_len-1):
            threshold = self._teacher_forcing_ratio(training_steps=tr_steps)
            current_input_word = targets[:, i] if random.uniform(0.05, 0.995) > threshold else self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)
            logprob = self.to_final_output(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long().cuda()
        seq_logProb, seq_predictions = [], []
        assumption_seq_len = 28
        
        for i in range(assumption_seq_len-1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)
            logprob = self.to_final_output(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def _teacher_forcing_ratio(self, training_steps):
        return expit(training_steps/20 + 0.85)

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, avi_feat, mode, target_sentences=None, tr_steps=None):
        encoder_outputs, encoder_last_hidden_state = self.encoder(avi_feat)
        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(encoder_last_hidden_state=encoder_last_hidden_state, encoder_output=encoder_outputs,
                                                        targets=target_sentences, mode=mode, tr_steps=tr_steps)
        else: 
            seq_logProb, seq_predictions = self.decoder.infer(encoder_last_hidden_state=encoder_last_hidden_state, encoder_output=encoder_outputs)
        return seq_logProb, seq_predictions


def compute_loss(loss_fn, predictions, targets, lengths):
    predict_cat, groundT_cat = None, None
    for batch in range(len(predictions)):
        predict = predictions[batch][:lengths[batch] - 1]
        ground_truth = targets[batch][:lengths[batch] - 1]
        predict_cat = predict if predict_cat is None else torch.cat((predict_cat, predict), dim=0)
        groundT_cat = ground_truth if groundT_cat is None else torch.cat((groundT_cat, ground_truth), dim=0)

    loss = loss_fn(predict_cat, groundT_cat)
    return loss / len(predictions)

def train_model(model, epochs, loss_fn, optimizer, train_loader):
    model.train()
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            avi_feats, ground_truths, lengths = batch
            avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()
            optimizer.zero_grad()
            seq_logProb, seq_predictions = model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)
            ground_truths = ground_truths[:, 1:]
            loss = compute_loss(loss_fn, seq_logProb, ground_truths, lengths)
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
        
def test_model(test_loader, model, i2w):
    model.eval() 
    results = []
    
    for batch_idx, batch in enumerate(test_loader):
        video_id, avi_feats = batch
        avi_feats = avi_feats.cuda()  
        avi_feats = Variable(avi_feats).float()

        seq_logProb, seq_predictions = model(avi_feats, mode='inference')
        
        
        predicted_captions = [[i2w[x.item()] if i2w[x.item()] != '<UNK>' else 'something' for x in s] for s in seq_predictions]
        predicted_captions = [' '.join(s).split('<EOS>')[0] for s in predicted_captions]

        batch_results = zip(video_id, predicted_captions)
        results.extend(batch_results)

    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    i2w, w2i, word_dict = preprocess_data()

    with open('i2w_yalluri.pickle', 'wb') as handle:
        pickle.dump(i2w, handle, protocol=pickle.HIGHEST_PROTOCOL)

    label_file = '/training_data/feat'
    files_dir = 'training_label.json'
    train_dataset = CustomTrainDataset(label_file, files_dir, word_dict, w2i)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=8, collate_fn=batch_data)

    epochs = 40
    encoder = Encoder()
    decoder = Decoder(512, len(i2w) + 4, len(i2w) + 4, 1024, 0.3)
    model = Seq2SeqModel(encoder=encoder, decoder=decoder).cuda()


    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

    for epoch in range(1):
        train_model(model, epochs, loss_fn, optimizer, train_dataloader)

    torch.save(model, 'model_yalluri.h5')
    print("Training complete")

if __name__ == "__main__":
    main()