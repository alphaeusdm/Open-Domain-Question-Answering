
import pandas as pd
from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch, gc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from datasets import load_metric
from sklearn.model_selection import train_test_split


def get_components(data):
    """
    get all the components from the data file.
    the components include context, questions, answers, start and end positions
    :param data: dataframe object
    :return: list of context, questions, answers, start positions and end positions
    """
    context = []
    questions = []
    answers = []
    start_index = []
    end_index = []
    for entry in data:
        con = entry['story']
        for id in range(len(entry['questions'])):
            if entry['answers'][id]['span_start'] >= 0:
                context.append(con)
                questions.append(entry['questions'][id]['input_text'])
                if entry['answers'][id]['span_text'][0] == " ":
                  entry['answers'][id]['span_text'] = entry['answers'][id]['span_text'][1:]
                  entry['answers'][id]['span_start'] = 1 + entry['answers'][id]['span_start']
                if entry['answers'][id]['span_text'][-1] == " ":
                  entry['answers'][id]['span_text'] = entry['answers'][id]['span_text'][:-1]
                  entry['answers'][id]['span_end'] = entry['answers'][id]['span_end'] - 1
                start_index.append(entry['answers'][id]['span_start'])
                end_index.append(entry['answers'][id]['span_end']-1)
                answers.append(entry['answers'][id]['span_text'])
                
    return questions, answers, start_index, end_index, context


def get_encodings(tokenizer, context, questions):
    """
    tokenize the questions and contexts
    :param tokenizer: bert tokenizer
    :param context: list of contexts
    :param questions: list of questions
    :return: tokenized inputs
    """
    return tokenizer(questions, context, truncation=True, padding=True)


def convert_to_ids(tokenizer, encodings, start_index, end_index):
    """
    get the start and end positions for the encodings.
    :param tokenizer: bert tokenizer
    :param encodings: tokenized inputs
    :param start_index: list of start positions
    :param end_index: list of end positions
    :return:
    """
    start_positions = []
    end_positions = []

    for i in range(len(start_index)):
        start_positions.append(encodings.char_to_token(i, start_index[i], sequence_index=1))
        end_positions.append(encodings.char_to_token(i, end_index[i], sequence_index=1))
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    # update the encodings with start and end positions.
    encodings.update({'start_positions':start_positions, 'end_positions':end_positions})


class Tdataset(torch.utils.data.Dataset):
    """
    Class to get the items from the encodings
    """
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, index):
        return {key : torch.tensor(value[index]) for key, value in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


def train(tokenizer, train_data, val_data, model):
    """
    Train the model
    :param tokenizer: bert tokenizer
    :param train_data: training data
    :param val_data: validation data
    :param model: bert model
    :return:
    """
    # check if gpu is available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.train()

    # initialize the Adam optimizer.
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # convert the training and validation to pytorch compatible format
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=True)

    # define number of epochs and initialize the progress bar
    epochs = 5
    steps = epochs * len(train_loader)
    scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=steps)
    progress_bar = tqdm(range(steps))

    # start training
    for epoch in range(epochs):
        print(epoch)
        for batch in train_loader:
            optimizer.zero_grad()
            # get items from the encoding dictionary to be passed as an input.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions)
            # get the total loss and perform back propagation
            loss = out[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress_bar.update(1)
        # save the model for each epoch
        torch.save(model, '/content/drive/MyDrive/model/model.pth')

    # test the model
    model.eval()
    evaluate(tokenizer, model, val_loader)

def evaluate(tokenizer, model, val_data):
    # evaluate the model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    metric = load_metric('accuracy')
    model.eval()
    count = 0
    for batch in val_data:
        print(count)
        inputs = batch['input_ids'].to(device)
        attention = batch["attention_mask"].to(device)
        token_ids = batch['token_type_ids'].to(device)
        start = batch['start_positions'].to(device)
        end = batch['end_positions'].to(device)
        with torch.no_grad():
          output = model(input_ids=inputs, attention_mask=attention, token_type_ids=token_ids)
        start_preds = torch.argmax(output.start_logits, dim=1)
        end_preds = torch.argmax(output.end_logits, dim=1)
        preds = torch.cat((start_preds, end_preds), 0)
        gold = torch.cat((start, end), 0)
        metric.add_batch(predictions=preds, references=gold)
        count += 1
    # compute accuracy and print it
    acc = metric.compute()
    print('accuracy : ', acc['accuracy'])

def main():
    # read training and validation data
    data = pd.read_json('/content/drive/MyDrive/model/coqa-train-v1.0.json')['data']
    data = data[0:1000]

    # split the data into train and test set.
    training_data, validation_data = train_test_split(data, test_size=0.2, shuffle=True)
    
    # initialize tokenizer and model from pretrained bert models
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    
    # get relevent components from the datasets
    train_questions, train_answers, train_start_index, train_end_index, train_context = get_components(training_data)
    val_questions, val_answers, val_start_index, val_end_index, val_context = get_components(validation_data)
    
    # get encodings
    train_encodings = get_encodings(tokenizer, train_context, train_questions)
    val_encodings = get_encodings(tokenizer, val_context, val_questions)
    
    # convert encodings to ids
    convert_to_ids(tokenizer, train_encodings, train_start_index, train_end_index)
    convert_to_ids(tokenizer, val_encodings, val_start_index, val_end_index)
    
    # convert encodings to datasets for training
    train_data = Tdataset(train_encodings)
    val_data = Tdataset(val_encodings)
    
    # fine-tune and evaluate the model
    train(tokenizer, train_data, val_data, model)

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    main()





