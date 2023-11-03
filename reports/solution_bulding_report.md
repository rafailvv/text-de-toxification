# Baseline: Dictionary-based Approach

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

# Hypothesis 1: LSTM model from scratch

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

# Hypothesis 2: Pre-trained BERT model

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

# Hypothesis 3: Tuning Text-To-Text Transfer Transformer (T5)

### Objective:
To harness a versatile text-to-text framework that could intake toxic sentences and output non-toxic counterparts, utilizing the ParaNMT-detox dataset for both detection and transformation of text.

### Methodology:
T5 was fine-tuned on the provided dataset, framing the task as translating toxic text (reference) to non-toxic text (translation). The model took into account not only the toxicity levels but also the similarity and length difference metrics to ensure the generated text was a faithful paraphrase of the original.

### Advantages:
- **Generative De-toxification:** T5's ability to generate text allowed for active transformation from toxic to non-toxic content, while aiming to maintain high similarity and appropriate length difference.
- **Dataset Maximization:** By utilizing all available data columns, T5's fine-tuning was highly targeted to the nuances of de-toxification as per the dataset's specifics.


### Limitations:

- **Complexity in Tuning:** Fine-tuning T5, especially for specific tasks like text de-toxification, requires careful balancing of parameters to avoid the model diverging or overfitting.
- **Generative Risks:** As a generative model, T5 might produce plausible-sounding but factually incorrect or nonsensical text if not guided properly by the training data and constraints.
- **Computation Demands:** T5's architecture is even larger than BERT's, requiring substantial computational power for both training and inference, which could be a bottleneck.
- **Data Representation:** T5 relies on the quality of the input data to understand the task at hand; if the dataset has biases or is not representative, the model's outputs will reflect these flaws.
- **Potential for Over-sanitization:** There is a risk that the model could over-sanitize the text, stripping away not just toxicity but also meaningful content or affecting the style and voice of the original author.

# Results

Upon evaluation, the following results were observed:

- **Baseline Model:** Showed improvement in identifying and mitigating clear-cut toxic phrases but struggled with nuanced language and context sensitivity.
- **LSTM Model:** Demonstrated a better grasp over context and a more dynamic response to toxicity levels, yet was overshadowed by the more advanced models in terms of accuracy and generation quality.
- **BERT Model:** Provided significant advancements in detecting contextual toxicity and was adept at identifying and understanding subtleties in language. However, it did not generate alternative text.
- **T5 Model:** Excelled in both identifying toxicity and producing high-quality, non-toxic paraphrases of the original text. It leveraged the dataset to its fullest, optimizing for similarity and length difference, and showed the highest scores in human evaluations.

The results indicate that while the baseline provided a fast and straightforward method for identifying blatantly toxic content, it was insufficient for complex tasks. The LSTM model offered better context understanding but required substantial computational effort. BERT excelled in toxicity detection but lacked generative capabilities. T5 emerged as the most comprehensive approach, adeptly balancing the detection and creation of non-toxic alternatives, proving its efficacy in handling the de-toxification task with the given dataset.
