## Yelp Reviews Sentiment Analysis using Large Language Models
The objective of this project is to perform multi-class classification on Yelp reviews to predict star ratings (1-5 stars) using Large Language Model concept. This project contains evaluation of difference of accurate predictions of **Classic Machine Learning (Logistic Regression/Random Forest/ Naive Bayes)** Versus **Large Langugage model Transfer Learning** concepts.

**Goals:**

- Classify Yelp reviews into one of five-star ratings (1, 2, 3, 4, or 5 stars).
- Analyze performance across different star categories to ensure balanced accuracy.
- Improve the model to handle imbalanced classes and provide reliable predictions.

## Dataset
**Dataset Name:** Yelp Review Dataset

<p align="center" margin-top="20px" margin-bottom="20px">
<img width="50%" src="https://github.com/hrmn-preet/llm-project/blob/main/images/dataset.png">
</img>
</p>

**Description:** The dataset contains a large number of reviews from Yelp, including text reviews and associated metadata such as star ratings. There are total of 5 class, equally balanced with data.

**Size:**
Total Reviews: 6,50,000 training rows and 50,000 testing rows.

**Distribution:** 

For training
- 1-Star / Class 0 - 130,000 rows
- 2-Stars / Class 1 - 130,000 rows
- 3-Stars / Class 2 - 130,000 rows
- 4-Stars / Class 3 - 130,000 rows
- 5-Stars / Class 4 - 130,000 rows

For testing
- 1-Star / Class 0 - 10,000 rows
- 2-Stars / Class 1 - 10,000 rows
- 3-Stars / Class 2 - 10,000 rows
- 4-Stars / Class 3 - 10,000 rows
- 5-Stars / Class 4 - 10,000 rows


**Star Rating:** The rating given by the user (1 to 5 stars).
Industry Category: The business category (e.g., Restaurants, Shopping, Health & Medical).

**Preprocessing:** 
- Removal of special characters, stopwords, and HTML tags, 
- Converting text into tokens compatible with the chosen model, 
- Adjusting review lengths to a fixed size for uniformity in input.

## Pre-trained Model
**Model Name:** *LiYuan/amazon-review-sentiment-analysis*

**Source:** Hugging Face Transformers library

**Pre-training:** The model has been pre-trained on a large corpus of text data to capture language patterns and context.
Fine-tuning:

The pre-trained model is fine-tuned on the Yelp Review dataset to adapt it for the specific task of multi-class classification.

## Performance Metrics
**Accuracy:** The percentage of correctly classified reviews.

**F1 Score:** Harmonic mean of precision and recall for each class, providing a balanced measure of performance.

**Confusion Matrix:** A matrix to visualize the performance across all star rating classes.

## Fine-tuned Model Evaluation Results

<p align="center" margin-top="20px" margin-bottom="20px">
<img src="https://github.com/hrmn-preet/llm-project/blob/main/images/Screenshot%202024-07-06%20223743.png">
</img>
</p>

## Hyperparameters

**Learning Rate:** The learning rate of 2e-5 was the best among range of 2e-5 to 5e-5. The higher learning rate did increase the speed of execution but it dropped accuracy.

**Batch Size:** The accuracy jumped when batch size was switched to 16.

**Epochs:** Model yielded better results with 2 epochs

**Warmup Steps:** The fine tuned model does not have any warm steps initialised but optimized model yielded best results on warmup steps of 200.

## Optimization:

**Learning Rate:** Adjusted the learning rate to balance between convergence speed and performance.

**Batch Size:** Experimented with different batch sizes to optimize GPU memory usage and training time.

**Epochs:** Monitored evaluation metrics across multiple epochs to prevent overfitting while ensuring sufficient training.

**Warmup Steps:** Used a warmup phase to gradually increase the learning rate at the beginning of training, helping the model to stabilize.

## Logistic Regression Results on manual tokenized dataset

The logistic regression gave 54.98% accuracy which is lower than pre-trained model. The tokenization and preprocessing was done manually.

<p align="center" margin-top="20px" margin-bottom="20px">
<img src="https://github.com/hrmn-preet/llm-project/blob/main/images/lr.png" width="50%">
</img>
</p>

## Relevant Links:

**The fine-tuned and optimized model**
<p align="center">
 <a href="https://huggingface.co/harmanpreet-kaur/yelp-review-sentiment-analysis-model-1">
<img width="80%" src="https://github.com/hrmn-preet/llm-project/blob/main/images/model.png">
</img>
</p>

**The Training and evaluation data can be found [here](https://huggingface.co/datasets/Yelp/yelp_review_full)** 
