"""Training the best model"""
import pandas as pd
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
import torch

# Load the dataset
data = pd.read_csv("../../data/interim/most_toxic_data.csv").head(10000)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Define the Dataset
class DetoxDataset(Dataset):
    """
    A dataset class for text detoxification tasks, wrapping around a given tokenizer and dataset.

    This class processes both toxic and non-toxic text pairs for training purposes, tokenizing
    and encoding them into a suitable format for machine learning models.

    Attributes:
        tokenizer: A tokenizer that is used for encoding texts.
        input_texts (list): A list of toxic reference texts.
        target_texts (list): A list of non-toxic translation texts.
        max_length (int): The maximum length of the tokenized output.
    """

    def __init__(self, text_tokenizer, corpus, max_length=512):
        """
        Initializes the DetoxDataset with a tokenizer, data, and an optional max_length.

        The input data is expected to be a dictionary with 'reference' and 'translation' keys,
        containing the toxic and non-toxic texts respectively.

        Args:
            text_tokenizer: The tokenizer used to encode the texts.
            corpus (dict): A dictionary containing 'reference' and 'translation' keys.
            max_length (int, optional): Maximum sequence length for tokenization. Defaults to 512.
        """
        self.tokenizer = text_tokenizer
        self.input_texts = corpus["reference"].tolist()  # Assuming this is the toxic text
        self.target_texts = corpus[
            "translation"
        ].tolist()  # Assuming this is the non-toxic text
        self.max_length = max_length

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.input_texts)

    def __getitem__(self, idx):
        """
        Retrieves an item by its index and returns a dictionary containing the tokenized
        and encoded versions of the source and target texts.

        The padding tokens in the target are replaced with -100
        to ignore them during loss calculation.

        Args:
            idx (int): The index of the data item to fetch.

        Returns:
            dict: A dictionary with 'input_ids', 'attention_mask', and 'labels' for the model.
        """
        source_text = self.input_texts[idx]
        target_text = self.target_texts[idx]

        # Tokenize the source text
        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize the target text
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        target_ids = target["input_ids"].squeeze()

        # The labels are the target_ids without the padding token
        labels = target_ids.masked_fill(target_ids == self.tokenizer.pad_token_id, -100)

        return {
            "input_ids": source_ids,
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels,
        }


# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# Create the dataset
dataset = DetoxDataset(tokenizer, data)

# Split the dataset
train_size = int(0.8 * len(dataset))
train_dataset, eval_dataset = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size]
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("../../model/detoxified_t5_model_generated")
tokenizer.save_pretrained("../../model/detoxified_t5_model_generated")
