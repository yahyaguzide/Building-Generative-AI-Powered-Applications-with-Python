from flask import Flask, request
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import json

app = Flask(__name__)
CORS(app)


model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []


@app.route('/')
def home():
    return 'Hello, World!'


@app.route('/chatbot',  methods=['POST'])
def handle_prompt() -> json:
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']

    history = "\n".join(conversation_history)

    inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")

    # max_length will acuse model to crash at some point as history grows
    outputs = model.generate(**inputs, max_length=60)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    conversation_history.append(input_text)
    conversation_history.append(response)

    if len(conversation_history) > 3:
        conversation_history.pop(0)

    return response


if __name__ == '__main__':
    app.run()
