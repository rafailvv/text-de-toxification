# Final Solution Report for the Text De-toxification Task

## Introduction
The text de-toxification project aimed to develop models capable of identifying and mitigating toxic content in textual data. A structured approach was adopted, involving data analysis, model specification, training, and evaluation, to create a solution that ensures the integrity and positivity of digital communication.

## Data Analysis
The data preparation phase involved standardizing numerical features and cleaning text data. Numerical features in the ParaNMT-detox dataset were standardized to have a mean of 0 and a standard deviation of 1. Text data was processed by converting the 'reference' and 'translation' columns to lowercase and removing special characters to improve data quality for model training.
<p float="left">
  <img src="https://github.com/rafailvv/text-de-toxification/blob/master/reports/figures/toxic_texts_word_cloud.png" width="48%" />
  <img src="https://github.com/rafailvv/text-de-toxification/blob/master/reports/figures/non-toxic_texts_word_cloud.png" width="48%" />
</p>
<p float="left">
  <img src="https://github.com/rafailvv/text-de-toxification/blob/master/reports/figures/distribution_length_differences.png" width="23%" />
  <img src="https://github.com/rafailvv/text-de-toxification/blob/master/reports/figures/distribution_reference_toxicity_scores.png" width="23%" />
  <img src="https://github.com/rafailvv/text-de-toxification/blob/master/reports/figures/distribution_similarity_scores.png" width="23%" />
  <img src="https://github.com/rafailvv/text-de-toxification/blob/master/reports/figures/distribution_translated_toxicity_scores.png" width="23%" />
</p>


## Model Specification
Several models were developed, each with unique specifications:

- A dictionary-based approach served as the baseline, utilizing a predefined set of criteria for identifying toxic phrases.
- A custom LSTM network was trained from scratch to learn textual contexts and mitigate toxicity.
- A pre-trained BERT model was fine-tuned to detect toxic content with improved contextual awareness.
- The Text-To-Text Transfer Transformer (T5) was fine-tuned for a dual role in detecting and transforming toxic sentences to non-toxic ones.

## Training Process
The models underwent rigorous training processes:

- The **dictionary-based** model was populated using the ParaNMT-detox corpus.
- The **LSTM** model was trained using sentence pairs to understand differences in toxicity levels.
- The **BERT** was fine-tuned with the ParaNMT-detox corpus to improve detection of toxic and non-toxic text pairs.
- The **T5** was specifically fine-tuned on 10000 rows, 10 epochs translating toxic sentences into non-toxic versions by ParaNMT-detox corpus on most toxic data.

## Evaluation
The models were evaluated using the **BLEU metric** for translation quality and a fine-tuned sentiment analysis model from [Hugging Face's repository](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) to ensure the preservation of sentiment post de-toxification.

## Results
The baseline model showed potential for clear-cut cases but lacked nuanced understanding. The LSTM model offered better context understanding but was computationally intensive. BERT excelled in detecting contextual toxicity but did not generate alternatives. T5 proved the most effective, achieving a balance between recognizing toxic content and producing quality, non-toxic versions. It demonstrated improvement over the baseline with a BLEU score of **24.47967691937549** and an accuracy of **0.468**, even with limited training data (10000 rows).

In conclusion, the **T5 model** emerged as the superior solution for the text de-toxification task, effectively balancing detection and creation of non-toxic text alternatives within the constraints of the dataset and computational resources.

The relatively low values in the BLEU score and accuracy for the T5 model can be attributed to the **constraints of computational resources and extended training time**. Despite these limitations, the model still outperformed its counterparts, highlighting its efficiency in handling the text de-toxification task with limited resources.

Also, an **application was created using PyQt6** to facilitate exploratory analysis and result-oriented visualizations. This application is designed to present the outcomes of the text de-toxification models in a more user-friendly and interactive manner, allowing for a deeper dive into the models' performances and the impacts of the training data and computational constraints on the results.
<p float="left">
  <img src="https://github.com/rafailvv/text-de-toxification/blob/master/reports/figures/application_results_tab.PNG" width="48%" />
  <img src="https://github.com/rafailvv/text-de-toxification/blob/master/reports/figures/application_results_tab.PNG" width="48%" />
</p>
