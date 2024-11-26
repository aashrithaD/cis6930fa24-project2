# cis6930fa24 -- Project2

Name: AASHRITHA REDDY DONAPATI

# Project Description 
This project aims to develop an "Unredactor" capable of predicting the most likely names hidden in a dataset containing redacted contexts. The dataset consists of movie reviews where some names have been obscured. By utilizing Natural Language Processing (NLP) techniques and machine learning, the project identifies these names based on the surrounding context. The workflow includes cleaning the data, extracting contextual features, training a Random Forest classifier, and assessing the model's performance using metrics such as precision, recall, and F1-score.

# Environment Setup

1. Clone the Repository:

    git clone <repository-url>
    cd <folder_name>

2. Install Dependencies:

    pip install -r 

3. Download Required Models
   Ensure you have the SpaCy model installed:

    python -m spacy download en_core_web_md

## How to run  

1. Prepare the Dataset:  
Ensure the input files (unredactor.tsv and test.tsv) are in the same directory as the script.  

2. To execute the script:  
pipenv run python <function_name>  

3. To run the test cases:  
pipenv run python -m pytest -v     

## Output:  
The cleaned file is saved as unredactor_cleaned.tsv.  
Validation metrics (accuracy, precision, recall, F1-score) are printed to the terminal.  
Predictions are saved to submissions.tsv in the required format.  

EXAMPLE:    

pipenv run python main.py    

input:   

1 But he must get past the princes and █████ first.    
2 His wife, scheming ██████████████, longs for the finer things in life.  

output:   

Cleaned file saved as unredactor_cleaned.tsv  
Accuracy: 0.0434  
Precision: 0.8755  
Recall: 0.0434  
F1 Score: 0.0394  
Predictions saved to submissions.tsv in the required format.  

submissions.tsv:     
1 Frankie Muniz    
2 Mr. Jon Keeyes    

## Functions

#### main.py

1. clean_input_file(file_path, output_path): Cleans the input file by ensuring that each line contains exactly three fields (split, name, and context).
```sh
Args:
file_path (str): The path to the input file (e.g., unredactor.tsv).
output_path (str): The path to save the cleaned file (e.g., unredactor_cleaned.tsv).

Return: None. The cleaned file is written to output_path.
```
2. add_context_features(df): Extracts features from the redacted context, including text before and after the redaction and redaction length.
```sh
Args:
df (pandas.DataFrame): Input DataFrame with a context column.

Return: (pandas.DataFrame) A DataFrame with additional columns:
redaction_length (int): Count of redacted characters (█).
text_before_redaction (str): Text before the redaction.
text_after_redaction (str): Text after the redaction.
combined_context (str): Concatenation of text_before_redaction and text_after_redaction.
```
3. extract_person_entities(text): Extracts named entities labeled as PERSON using the SpaCy NLP model.
```sh
Args:
text (str): Input text to extract named entities from.

Return: (list) A list of names (str) identified as PERSON.
```
4. add_named_entity_features(df): Adds a column of named entities extracted from the combined_context column using extract_person_entities.
```sh
Args:
df (pandas.DataFrame): Input DataFrame with a combined_context column.

Return: (pandas.DataFrame) A DataFrame with an additional column:
extracted_entities (list): List of names extracted from the context.
```
5. load_dataset(file_path): Loads and preprocesses the dataset by adding contextual and named entity features.
```sh
Args:
file_path (str): Path to the cleaned dataset file (e.g., unredactor_cleaned.tsv).

Return: (pandas.DataFrame) Preprocessed dataset with additional features:
redaction_length
text_before_redaction
text_after_redaction
combined_context
extracted_entities
```
6. train_name_prediction_model(training_df): Trains a machine learning model to predict redacted names using a Random Forest classifier.
```sh
Args:
training_df (pandas.DataFrame): Training data with combined_context and extracted_entities.

Return:
tfidf_vectorizer (TfidfVectorizer): Vectorizer used to transform text into numerical features.
rf_model (RandomForestClassifier): Trained Random Forest classifier.
```
7. evaluate_name_prediction_model(tfidf_vectorizer, rf_model, validation_df): Evaluates the model on the validation dataset and prints metrics.
```sh
Args:
tfidf_vectorizer (TfidfVectorizer): The vectorizer used to preprocess text.
rf_model (RandomForestClassifier): Trained model to evaluate.
validation_df (pandas.DataFrame): Validation data.

Return: None. Prints evaluation metrics:
Accuracy
Precision
Recall
F1 Score
```
8. predict_name_from_context(context, tfidf_vectorizer, rf_model): Predicts the redacted name based on the provided context.
```sh
Args:
context (str): Input text containing a redaction.
tfidf_vectorizer (TfidfVectorizer): Vectorizer for text transformation.
rf_model (RandomForestClassifier): Trained model for prediction.

Return: (str) Predicted name for the redaction.
```
9. save_name_predictions(test_df, tfidf_vectorizer, rf_model, output_file): Generates predictions for the test dataset and saves them to a file.
```sh
Args:
test_df (pandas.DataFrame): Test data with id and context columns.
tfidf_vectorizer (TfidfVectorizer): Vectorizer for text transformation.
rf_model (RandomForestClassifier): Trained model for prediction.
output_file (str): Path to save the predictions (e.g., submissions.tsv).

Return: None. Saves the predictions in the required format:
Columns: id (int) and name (str).
```
### test_redactor.py

1. test_clean_input_file: Verifies that the input file is cleaned by removing invalid lines and formatting correctly.

2. test_add_context_features: Checks that contextual features are correctly extracted and added to the DataFrame.

3. test_extract_person_entities: Ensures names (PERSON entities) are extracted correctly from text using SpaCy.

4. test_add_named_entity_features: Validates the addition of named entity features to the DataFrame.

5. test_train_name_prediction_model: Confirms the model is trained successfully and returns a vectorizer and classifier.

6. test_predict_name_from_context: Tests that the model predicts the correct name from a given redacted context.

## Bugs and Assumptions

Assumptions:

1. Each redaction corresponds to a single person’s name, whether it is a first name, last name, or full name.
2. The text before and after the redaction provides sufficient context for inference.
3. The redaction symbols (█) accurately reflect the length and position of the hidden name.
4. The SpaCy en_core_web_sm model is considered sufficient for identifying person entities in movie review texts.
5. There are no duplicate or overlapping redactions within the same context.
6. The test.tsv dataset is assumed to be well-structured and consistent with the expected schema.
7. The Random Forest classifier is assumed to be a suitable machine learning model for this task based on the extracted features.
8. The names to be predicted are assumed to exist in the training dataset; the model does not need to infer entirely unseen names.

Bugs:

1. Non-standard or improperly formatted contexts may lead to pipeline errors.
2. The model is prone to overfitting when training data is limited or imbalanced.
3. SpaCy’s NER model might inaccurately detect person entities in complex or domain-specific text.
4. The redaction length represented by █ symbols may not always correspond to the actual name length.
5. Name matching may fail due to differences in case sensitivity.
6. Empty or missing context fields are not handled and could disrupt preprocessing.
7. The pipeline assumes uniform formatting; irregular inputs may result in errors.
8. Contexts with minimal or unclear information may lead to inaccurate predictions.
9. Longer texts might result in reduced performance due to incomplete context handling.





