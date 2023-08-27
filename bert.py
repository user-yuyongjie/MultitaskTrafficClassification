import torch
from transformers import BertModel, BertTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
from torch import nn

# Load BERT model from HuggingFace
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

class MyDataset(Dataset):
    def __init__(self, sequences, labels, masks):
        self.sequences = sequences
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return dict(
            sequence=self.sequences[idx],
            input_ids=self.sequences[idx].tolist(),
            label=torch.tensor(self.labels[idx]),
            mask=torch.tensor(self.masks[idx])
        )

# Convert your data to PyTorch tensors and then create a Dataset and DataLoader for them
train_sequences = torch.tensor(trainData).long()
train_labels = torch.tensor(Y_train3).long()
train_masks = torch.tensor(trainmask).long()

val_sequences = torch.tensor(valData).long()
val_labels = torch.tensor(Y_val3).long()
val_masks = torch.tensor(valmask).long()

test_sequences = torch.tensor(testData).long()
test_labels = torch.tensor(Y_test3).long()
test_masks = torch.tensor(testmask).long()

# Create data loaders
batch_size = 64
train_data = MyDataset(train_sequences, train_labels, train_masks)
train_loader = DataLoader(train_data, batch_size=batch_size)

val_data = MyDataset(val_sequences, val_labels, val_masks)
val_loader = DataLoader(val_data, batch_size=batch_size)

test_data = MyDataset(test_sequences, test_labels, test_masks)
test_loader = DataLoader(test_data, batch_size=batch_size)

class MyModel(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(MyModel, self).__init__()
        self.bert = bert_model
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        out = self.out(outputs.pooler_output)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a new model with BERT
model = MyModel(bert_model, num_classes=5)
model = model.to(device)

# Define optimizer and criterion
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        input_ids = data['input_ids'].to(device)
        attention
        attention_mask = data['mask'].to(device)
        labels = data['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    total_loss, total_accuracy = 0, 0
    for data in val_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['mask'].to(device)
        labels = data['label'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1).flatten()
        accuracy = (preds == labels).cpu().numpy().mean() * 100
        total_accuracy += accuracy

    avg_loss = total_loss / len(val_loader)
    avg_acc = total_accuracy / len(val_loader)

    print(f"Validation Loss : {avg_loss}")
    print(f"Validation Accuracy : {avg_acc}%")

# Test
model.eval()
total_loss, total_accuracy = 0, 0
for data in test_loader:
    input_ids = data['input_ids'].to(device)
    attention_mask = data['mask'].to(device)
    labels = data['label'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    loss = criterion(outputs, labels)
    total_loss += loss.item()

    preds = torch.argmax(outputs, dim=1).flatten()
    accuracy = (preds == labels).cpu().numpy().mean() * 100
    total_accuracy += accuracy

avg_loss = total_loss / len(test_loader)
avg_acc = total_accuracy / len(test_loader)

print(f"Test Loss : {avg_loss}")
print(f"Test Accuracy : {avg_acc}%")