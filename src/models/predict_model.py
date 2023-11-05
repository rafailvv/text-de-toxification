"""Best model prediction"""
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch


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


def is_toxic(text, comments=False):
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
    results = classifier(text)

    for result in results:
        label = result["label"]
        score = result["score"]

        # If comments is True, print the label and score.
        if comments:
            print(f"Label: {label}, Score: {score}")


# Check if CUDA is available and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the model and tokenizer
MODEL_PATH = "../../models/detoxified_t5_model"
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)


# Initialize the classifier pipeline
classifier = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)

while True:
    # Get a toxic sentence from the user
    toxic_text = input("Write a toxic sentence: ")

    # Detoxify the input text
    text_to_analyze = detoxify_text(toxic_text)
    print(text_to_analyze)

    # Analyze the detoxified text for toxicity and print the results
    is_toxic(text_to_analyze, comments=True)
