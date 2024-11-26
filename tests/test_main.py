import pytest
import pandas as pd
import tempfile
from main import (
    clean_input_file,
    add_context_features,
    extract_person_entities,
    add_named_entity_features,
    load_dataset,
    train_name_prediction_model,
    predict_name_from_context,
)

def test_clean_input_file():
    raw_data = "training\tJohn Ritter\tThis is a test ██████ context.\ninvalid line\n"
    expected_cleaned_data = "training\tJohn Ritter\tThis is a test ██████ context.\n"

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_input:
        temp_input.write(raw_data)
        temp_input.seek(0)
        temp_input_path = temp_input.name

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_output:
        temp_output_path = temp_output.name

    clean_input_file(temp_input_path, temp_output_path)

    with open(temp_output_path, 'r') as cleaned_file:
        assert cleaned_file.read() == expected_cleaned_data

def test_add_context_features():
    df = pd.DataFrame({
        'split': ['training'],
        'name': ['John Ritter'],
        'context': ['This is a test ██████ context.']
    })
    processed_df = add_context_features(df.copy())
    assert 'redaction_length' in processed_df.columns
    assert 'text_before_redaction' in processed_df.columns
    assert 'text_after_redaction' in processed_df.columns
    assert 'combined_context' in processed_df.columns
    assert processed_df['redaction_length'].iloc[0] == 6

def test_extract_person_entities():
    entities = extract_person_entities("John Ritter is a person.")
    assert "John Ritter" in entities

def test_add_named_entity_features():
    df = pd.DataFrame({
        'split': ['training'],
        'name': ['John Ritter'],
        'context': ['This is a test ██████ context.']
    })

    processed_df = add_context_features(df.copy())

    processed_df = add_named_entity_features(processed_df)

    assert 'extracted_entities' in processed_df.columns
    assert isinstance(processed_df['extracted_entities'].iloc[0], list)

def test_train_name_prediction_model():
    df = pd.DataFrame({
        'split': ['training'],
        'name': ['John Ritter'],
        'context': ['This is a test ██████ context.']
    })
    processed_df = add_context_features(df.copy())
    processed_df = add_named_entity_features(processed_df)
    tfidf_vectorizer, rf_model = train_name_prediction_model(processed_df)
    assert tfidf_vectorizer is not None
    assert rf_model is not None

def test_predict_name_from_context():
    df = pd.DataFrame({
        'split': ['training'],
        'name': ['John Ritter'],
        'context': ['This is a test ██████ context.']
    })
    processed_df = add_context_features(df.copy())
    processed_df = add_named_entity_features(processed_df)
    tfidf_vectorizer, rf_model = train_name_prediction_model(processed_df)
    prediction = predict_name_from_context("This is a test ██████ context.", tfidf_vectorizer, rf_model)
    assert prediction == "John Ritter"
