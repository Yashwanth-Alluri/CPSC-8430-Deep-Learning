import sys
import torch
import json
from train_yalluri import CustomTestDataset, test_model, Seq2SeqModel, Encoder, Decoder, AttentionMechanism
from torch.utils.data import DataLoader
from data.bleu_eval import BLEU
import pickle

model = torch.load('model_yalluri.h5', map_location=lambda storage, loc: storage)

test_filepath = 'data/testing_data/feat'

test_dataset = CustomTestDataset(test_filepath)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)

with open('i2w_yalluri.pickle', 'rb') as handle:
    i2w = pickle.load(handle)

model = model.cuda()

inference_results = test_model(test_loader, model, i2w)

with open(sys.argv[2], 'w') as f:
    for video_id, caption in inference_results:
        f.write('{},{}\n'.format(video_id, caption))

test_labels = json.load(open('data/testing_label.json'))
output_file = sys.argv[2]

generated_captions = {}
with open(output_file, 'r') as f:
    for line in f:
        line = line.rstrip()
        comma_index = line.index(',')
        video_id = line[:comma_index]
        generated_caption = line[comma_index+1:]
        generated_captions[video_id] = generated_caption

bleu_scores = []
for item in test_labels:
    video_scores = []
    reference_captions = [caption.rstrip('.') for caption in item['caption']]
    video_scores.append(BLEU(generated_captions[item['id']], reference_captions, True))
    bleu_scores.append(video_scores[0])

average_bleu = sum(bleu_scores) / len(bleu_scores)
print()
print("Average BLEU score is " + str(average_bleu))
print()