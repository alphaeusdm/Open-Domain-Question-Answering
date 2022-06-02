from flask import Flask, request, redirect, jsonify, url_for
from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

@app.route('/')
def welcome():
    """
    create home page
    :return:
    """
    return redirect('/apidocs')

@app.route('/predict')
def get_answer():


    """Pass the Question and Context as input
    ---
    parameters:
        - name: question
          in: query
          type: string
          required: true
        - name: context
          in: query
          type: string
          required: true
    responses:
          200:
              description: The answer

    """
    question = request.args.get('question')
    context = request.args.get('context')

    # tokenize the question and context
    encoding = tokenizer(question, context, padding=True, truncation=True)
    model.eval()

    # get inputs to be given to the model
    input_ids = torch.tensor([encoding['input_ids']])
    token_type_ids = torch.tensor([encoding['token_type_ids']])
    attention_mask = torch.tensor([encoding['attention_mask']])

    # get predictions
    with torch.no_grad():
        out = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    # get start and end positions from the predictions
    start = torch.argmax(out.start_logits)
    end = torch.argmax(out.end_logits)

    # get back the tokens of the input
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])

    # get the answer
    answer = " "
    if start < end:
        answer = tokens[start]
        for i in range(start+1, end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
    return answer


if __name__ == '__main__':
    # initialize bert tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # check if gpu is available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model
    model = torch.load('model.pth', map_location=device)
    app.run(debug=True, host='0.0.0.0')
