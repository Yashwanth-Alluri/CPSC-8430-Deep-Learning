import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "AshtonIsNotHere/albert-large-v2-spoken-squad"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
qa_model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)

learning_rate = 2e-5
train_batch_size = 16
eval_batch_size = 64
seed = 42
num_epochs = 1
optimizer = AdamW(qa_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
scaler = GradScaler()
def parse_data(filepath):
    with open(filepath, 'r') as file:
        raw_data = json.load(file)
    contexts, queries, answers = [], [], []
    for section in raw_data['data']:
        for para in section['paragraphs']:
            context_text = para['context'].lower()
            for qa_item in para['qas']:
                question_text = qa_item['question'].lower()
                for answer in qa_item['answers']:
                    contexts.append(context_text)
                    queries.append(question_text)
                    answers.append({
                        'text': answer['text'].lower(),
                        'start': answer['answer_start'],
                        'end': answer['answer_start'] + len(answer['text'])
                    })
    return contexts, queries, answers

train_contexts, train_questions, train_labels = parse_data('Spoken-SQuAD/spoken_train-v1.1.json')
valid_contexts, valid_questions, valid_labels = parse_data('Spoken-SQuAD/spoken_test-v1.1.json')

train_data = tokenizer(train_contexts, train_questions, padding=True, truncation=True, max_length=512)
valid_data = tokenizer(valid_contexts, valid_questions, padding=True, truncation=True, max_length=512)

class QADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        enc = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        enc.update({'start_positions': torch.tensor(self.labels[idx]['start']),
                    'end_positions': torch.tensor(self.labels[idx]['end'])})
        return enc

    def __len__(self):
        return len(self.labels)

train_dataset = QADataset(train_data, train_labels)
valid_dataset = QADataset(valid_data, valid_labels)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=eval_batch_size)

total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def train_epoch(model, data_loader, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        with autocast():
            outputs = model(
                input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device),token_type_ids=batch['token_type_ids'].to(device),start_positions=batch['start_positions'].to(device),end_positions=batch['end_positions'].to(device)
            )
            loss = outputs.loss
            total_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader):
    model.eval()
    predicted, actual = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                token_type_ids=batch['token_type_ids'].to(device)
            )
            start_logits, end_logits = outputs.start_logits, outputs.end_logits
            
            for i in range(start_logits.size(0)):
                start_idx = torch.argmax(start_logits[i]).item()
                end_idx = torch.argmax(end_logits[i]).item()
                
                pred_text = tokenizer.decode(batch['input_ids'][i][start_idx:end_idx+1], skip_special_tokens=True)
                actual_text = tokenizer.decode(batch['input_ids'][i][batch['start_positions'][i]:batch['end_positions'][i]+1], skip_special_tokens=True)
                
                predicted.append(pred_text)
                actual.append(actual_text)

    return predicted, actual

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train_epoch(qa_model, train_loader, optimizer, scheduler, scaler)
    predictions, references = evaluate(qa_model, valid_loader)

    epoch_f1 = f1_score([pred == ref for pred, ref in zip(predictions, references)], [True] * len(predictions))
    print(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}, F1 Score: {epoch_f1:.2f}")
