import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Cleaning Function
def clean_input_file(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            fields = line.strip().split('\t')
            if len(fields) == 3:
                outfile.write(line)
    print(f"Cleaned file saved as {output_path}")

# Preprocessing Function
def add_context_features(df):
    df['redaction_length'] = df['context'].apply(lambda x: x.count('█'))
    df['text_before_redaction'] = df['context'].apply(lambda x: x.split('█')[0].strip() if '█' in x else '')
    df['text_after_redaction'] = df['context'].apply(lambda x: x.split('█')[-1].strip() if '█' in x else '')
    df['combined_context'] = df['text_before_redaction'] + ' ' + df['text_after_redaction']
    return df

# Extract Named Entities using spaCy
def extract_person_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

# Prepare Features Using spaCy
def add_named_entity_features(df):
    df['extracted_entities'] = df['combined_context'].apply(lambda x: extract_person_entities(x))
    return df

# Load Dataset
def load_dataset(file_path):
    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            names=['split', 'name', 'context'],
            on_bad_lines='skip'  # Skip problematic rows
        )
        df = add_context_features(df)
        df = add_named_entity_features(df)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Train the Model
def train_name_prediction_model(training_df):
    training_df.loc[:, 'feature_text'] = training_df['combined_context'] + " " + training_df['extracted_entities'].apply(lambda x: " ".join(x))
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train = tfidf_vectorizer.fit_transform(training_df['feature_text'])
    y_train = training_df['name']
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return tfidf_vectorizer, rf_model

# Evaluate the Model
def evaluate_name_prediction_model(tfidf_vectorizer, rf_model, validation_df):
    validation_df.loc[:, 'feature_text'] = validation_df['combined_context'] + " " + validation_df['extracted_entities'].apply(lambda x: " ".join(x))
    X_val = tfidf_vectorizer.transform(validation_df['feature_text'])
    y_val = validation_df['name']
    
    val_predictions = rf_model.predict(X_val)
    
    accuracy = accuracy_score(y_val, val_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, val_predictions, average='weighted', zero_division=1)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Prediction Function
def predict_name_from_context(context, tfidf_vectorizer, rf_model):
    doc = nlp(context)
    entities = " ".join([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
    feature_text = context + " " + entities
    features = tfidf_vectorizer.transform([feature_text])
    return rf_model.predict(features)[0]

# Save Predictions to File
def save_name_predictions(test_df, tfidf_vectorizer, rf_model, output_file):
    predictions = []
    for context in test_df['context']:
        predicted_name = predict_name_from_context(context, tfidf_vectorizer, rf_model)
        predictions.append(predicted_name)
    
    output_df = pd.DataFrame({
        'id': test_df['id'], 
        'name': predictions
    })
    
    output_df.to_csv(output_file, sep='\t', index=False)
    print(f"Predictions saved to {output_file} in the required format.")

# Main
if __name__ == "__main__":
    input_file = 'unredactor.tsv'
    cleaned_file = 'unredactor_cleaned.tsv'
    
    clean_input_file(input_file, cleaned_file)
    
    dataset = load_dataset(cleaned_file)
    test_dataset = pd.read_csv('test.tsv', sep='\t', header=None, names=['id', 'context'])

    if dataset is not None:
        training_data = dataset[dataset['split'] == 'training'].copy()
        validation_data = dataset[dataset['split'] == 'validation'].copy()
        
        tfidf_vectorizer, rf_model = train_name_prediction_model(training_data)
        evaluate_name_prediction_model(tfidf_vectorizer, rf_model, validation_data)
        
        save_name_predictions(test_dataset, tfidf_vectorizer, rf_model, "submissions.tsv")
