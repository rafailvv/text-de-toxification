"""Scripts to create exploratory and results oriented visualizations"""
import torch
from PyQt6.QtWidgets import QApplication
from PyQt6 import uic
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

def detoxify_text(text):
    """
    This function takes a piece of text and runs it through a model to detoxify it.
    The model will attempt to rephrase the input text in a way that removes any toxic
    content, while preserving the original meaning as much as possible.

    :param text: A string containing the text to detoxify.
    :return: A string containing the detoxified version of the text.
    """
    model.eval()  # Ensure the model is in evaluation mode
    # Tokenize the input text and prepare it for the model
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    # Generate output using the model. 'num_beams' sets the number of beams for beam search.
    outputs = model.generate(
        input_ids, max_length=512, num_beams=5, early_stopping=True
    )
    # Decode the output tokens to a string and skip special tokens like padding, etc.
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def is_toxic(text):
    """
    This function uses a classifier to determine if the given text contains toxic content.
    It prints out the labels and scores if the 'comments' parameter is set to True.

    :param text: A string containing the text that needs to be analyzed for toxicity.
    :param comments: A boolean flag that determines whether to print the results.
                     If True, the function will print out the labels and scores
                     for each piece of text analyzed.
    :return: None
    """
    # Assume 'classifier' is a pre-defined function or model that can classify the toxicity.
    # Initialize the classifier pipeline
    classifier = pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    results = classifier(text)

    for result in results:
        label = result["label"]
        score = result["score"]
        return f"Label: {label}, Score: {score}"


def detoxification_pressed():
    """
    Method execute when button detoxification pressed
    :return:
    """
    text = form.lineEdit_toxic.text()
    form.label_toxity_result.setText("Converting...")
    detox_text = detoxify_text(text)
    form.lineEdit_non_toxic.setText(detox_text)
    result = is_toxic(detox_text)
    form.label_toxity_result.setText(
        f"The result of the toxicity analysis of the non-toxic sentence:\n{result}")


Form, Windows = uic.loadUiType('design.ui')
win = QApplication([])
windows = Windows()
form = Form()
form.setupUi(windows)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MODEL_PATH = "../../models/detoxified_t5_model"
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

form.detoxification_button.clicked.connect(detoxification_pressed)

windows.show()
win.exec()
