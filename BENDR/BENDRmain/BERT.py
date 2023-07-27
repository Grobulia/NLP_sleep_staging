#import transformers

from transformers import BertForPreTraining, BertConfig
from torch import nn, optim
from prepareInputForBERT import *
from torchmetrics import Accuracy


epochs = 1
lr = 1e-4


class InputDatasetForTransformer(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
        print(self.inputs.items())

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.inputs.items()}

    def __len__(self):
        return len(self.inputs['input_ids'])

inputDataset = InputDatasetForTransformer(input)
loader = torch.utils.data.DataLoader(inputDataset, batch_size=batch_size, shuffle=True)

config = BertConfig(vocab_size=750)

model = BertForPreTraining(config)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

model_type = f'BERT_b{batch_size}_lr{lr}_1874'
for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optimizer.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        next_sentence_label = batch['next_sentence_label'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        next_sentence_label=next_sentence_label,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optimizer.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        
    print('saving the model...')
    torch.save(model.state_dict(), f'{absolute_root}/saved_weights_{model_type}.pt')
    torch.save(model.state_dict(), f'{absolute_root}/saved_weights_{model_type}.h5')



