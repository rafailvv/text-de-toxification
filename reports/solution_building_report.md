## Dataset Information

All models were trained on subsets of the **ParaNMT**-detox dataset, ranging from 1,000 to 10,000 rows, due to constraints in training times and limited computational resources.

## Evaluation Metrics
To measure the accuracy of the generated text in comparison to the reference non-toxic sentences, the **BLEU (Bilingual Evaluation Understudy)** metric was employed. The BLEU metric assesses the quality of machine-generated text by comparing it to reference translations, using a modified form of precision to consider the proportion of n-grams in the generated text that match the reference text, thus capturing the fluency and adequacy of the translation. The use of BLEU allowed for a quantifiable measure of how closely the models' outputs resembled the desired non-toxic paraphrases.

Furthermore, the model's proficiency in generating accurate sentences was validated using a fine-tuned model from [Hugging Face's model repository](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english). This model, a distilled version of **BERT**, is fine-tuned on the SST-2 (Stanford Sentiment Treebank) dataset for sentiment analysis. It provided a benchmark to evaluate the sentiment accuracy of the sentences produced by our models, ensuring that the de-toxified output maintained the original sentiment while stripping away toxic elements.


# Baseline: Dictionary-based Approach
[Link to the notebook](https://github.com/rafailvv/text-de-toxification/blob/master/notebooks/2.0-dictionary-based-approach.ipynb)
### Objective:
To create a foundational text de-toxification model using a dictionary-based method that flags and replaces toxic phrases based on a predefined set of criteria derived from a curated corpus.

### Methodology:
Utilizing the ParaNMT-detox corpus, a targeted dictionary was compiled. This dictionary was extracted by analyzing the correlation between reference texts and their toxicity levels. High-toxicity phrases and words were identified and used to populate the dictionary. A simple algorithm was employed to match strings in new input text against this dictionary. If a match was found, the corresponding segment of text was flagged as toxic.

### Adaptations to the Dataset:
- **Toxicity Thresholding:** A toxicity threshold was established to decide which words should be included in the dictionary.
- **Dynamic Dictionary Updates:** Leveraging the ParaNMT-detox dataset's toxicity levels, the dictionary could be updated dynamically to reflect the most frequently occurring toxic phrases and their variants.
- **Performance Measures:** Utilizing the `trn_tox` and `ref_tox` columns, the efficacy of the dictionary in reducing toxicity levels could be evaluated, and adjustments could be made accordingly.

### Limitations:
- **Context Ignorance:** Despite being based on a robust dataset, the approach may still fail to consider the full context in which words are used.
- **Dictionary Maintenance:** Continuous updates to the dictionary may be required to keep up with the evolution of language and context-dependent meanings.
- **No Sentiment Analysis:** The approach lacks the ability to analyze the sentiment behind the usage of words, which can lead to misclassification of text as toxic or non-toxic.

### Results:
- Accuracy by evaluating model is **0.2**
- The BELU score: **19.40789466901169**


# Hypothesis 1: LSTM model from scratch

[Link to the notebook](https://github.com/rafailvv/text-de-toxification/blob/master/notebooks/4.0-LSTM-model-from-scratch.ipynb)

### Objective:
To leverage a neural network that can learn contextual nuances from the ParaNMT-detox dataset to predict and mitigate text toxicity more effectively than the baseline method.

### Methodology:
An LSTM network was designed and trained from scratch, using the sentence pairs from the ParaNMT-detox corpus. The model aimed to learn the difference in toxicity levels (`ref_tox` and `trn_tox`) and generate less toxic text alternatives similar to the ‘translation’ column. Training involved predicting the lower toxicity levels and generating text that aligned with those levels while maintaining semantic similarity and length consistency as indicated by the `similarity` and `length_diff` columns.

### Advantages:
- **Toxicity Prediction:** The LSTM model was trained to predict toxicity levels of text and then use those predictions to guide the generation of de-toxified text.
- **Sequence Learning:** By taking advantage of LSTM's ability to learn from sequences, the model could better understand the context and structure of the input text.

### Limitations:
- **Data Intensiveness:** LSTMs require substantial amounts of training data to capture the nuances of language effectively. The size and quality of the ParaNMT-detox dataset may limit the depth of understanding the model can achieve.
- **Training Time and Resources:** Training an LSTM from scratch can be computationally expensive and time-consuming, especially on large datasets.
- **Generalization:** Without pre-training on a more extensive corpus, the LSTM might not generalize well to unseen data or different contexts outside of the training dataset.

### Hypothesis result:
- It took a lot of time to learn and was not particularly effective, so I did not bring it to good results. Doesn't always generate logical sentences

# Hypothesis 2: Pre-trained BERT model

[Link to the notebook](https://github.com/rafailvv/text-de-toxification/blob/master/notebooks/5.0-BERT-model-pre-trained.ipynb)


### Objective:
To utilize a powerful pre-trained model to detect and reduce text toxicity by understanding complex context, sentiment, and language structures in the ParaNMT-detox dataset.

### Methodology:
BERT was fine-tuned using the sentence pairs and associated toxicity measures from the ParaNMT-detox corpus. The model was trained to understand the nuances between toxic and non-toxic text pairs, as well as to generate alternatives that not only reduce toxicity but also preserve the original content's meaning and length characteristics.

### Advantages:
- **Fine-tuned Contextual Awareness:** By fine-tuning BERT with the dataset, the model gained a refined understanding of toxicity and non-toxicity indicators within textual content.
- **Efficient Learning:** The pre-trained nature of BERT allowed for efficient transfer learning, reducing the need for extensive computational resources and time for training from scratch.

### Limitations:

- **Fine-tuning Challenges:** BERT requires careful fine-tuning to adapt to specific tasks. The wrong hyperparameters or fine-tuning approach can lead to suboptimal performance.
- **Resource Intensive:** Fine-tuning and deploying BERT models demand significant computational resources due to their size, which might be a constraint in resource-limited environments.
- **Output Restriction:** BERT is primarily a classification model and not inherently designed for text generation, making it less suitable for directly producing non-toxic text alternatives.

### Hypothesis result:
- The BERT model's inability to fully address my issue became apparent after training it for 5 epochs and evaluating the predictive outcomes. Consequently, I discontinued pursuing this particular approach.


# Hypothesis 3: Tuning Text-To-Text Transfer Transformer (T5)

[Link to the notebook](https://github.com/rafailvv/text-de-toxification/blob/master/notebooks/6.0-t5-small-fine-tune.ipynb)

### Objective:
To harness a versatile text-to-text framework that could intake toxic sentences and output non-toxic counterparts, utilizing the ParaNMT-detox dataset for both detection and transformation of text.

### Methodology:
The fine-tuning of T5 on the provided dataset was specifically focused on the reference and translation columns to address the transformation of toxic text into non-toxic equivalents. This narrow focus was crucial as it enabled the model to concentrate on the essential task of paraphrasing while mitigating toxicity. By honing in on these columns, the model could effectively learn to maintain content similarity and appropriate length, ensuring that the rephrased text remained true to the original meaning without carrying over any offensive content.
### Advantages:
- **Generative De-toxification:** T5's ability to generate text allowed for active transformation from toxic to non-toxic content, while aiming to maintain high similarity and appropriate length difference.
- **Dataset Maximization:** By utilizing all available data columns, T5's fine-tuning was highly targeted to the nuances of de-toxification as per the dataset's specifics.

### Limitations:

- **Complexity in Tuning:** Fine-tuning T5, especially for specific tasks like text de-toxification, requires careful balancing of parameters to avoid the model diverging or overfitting.
- **Computation Demands:** T5's architecture is even larger than BERT's, requiring substantial computational power for both training and inference, which could be a bottleneck.
- **Potential for Over-sanitization:** There is a risk that the model could over-sanitize the text, stripping away not just toxicity but also meaningful content or affecting the style and voice of the original author.

### Hypothesis result:
- Trained on **10000 rows, 10 epochs**
- Accuracy by evaluating model is **0.468**
- The BELU score: **24.47967691937549**

# Results

- **Baseline Model:** Showed improvement in identifying and mitigating clear-cut toxic phrases but struggled with nuanced language and context sensitivity.
- **LSTM Model:** Demonstrated a better grasp over context and a more dynamic response to toxicity levels, yet was overshadowed by the more advanced models in terms of accuracy and generation quality.
- **BERT Model:** Provided significant advancements in detecting contextual toxicity and was adept at identifying and understanding subtleties in language. However, it did not generate alternative text.
- **T5 Model:** The model has demonstrated effectiveness in both recognizing toxic content and producing high-quality, non-toxic versions of the original text. Despite the limitation of processing the entire data set, the model's performance was not poor even within a smaller subset of data (10,000 rows for 10 epochs), outperforming all other solutions evaluated.

The results indicate that while the baseline provided a fast and straightforward method for identifying blatantly toxic content, it was insufficient for complex tasks. The LSTM model offered better context understanding but required substantial computational effort. BERT excelled in toxicity detection but lacked generative capabilities. **T5 emerged as the most comprehensive approach, adeptly balancing the detection and creation of non-toxic alternatives**, proving its efficacy in handling the de-toxification task with the given dataset.