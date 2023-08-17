from flask import Flask, render_template, request
import tensorflow as tf
import pandas as pd
import logging
import sys
import os

from spacy.lang.en import English
from sklearn.preprocessing import LabelEncoder

# Current directory
current_dir = os.path.dirname(__file__)

app = Flask(__name__, static_folder='static', template_folder='template')

# Logging
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

# Make function to split sentences into characters
def split_chars(text):
    return " ".join(list(text))

# Create a function to read the lines of a document
def get_lines(filename):
    """
    Reads filename (a text filename) and returns the lines of text as a list.

    Args:
        filename: a string containing the target filepath.

    Returns:
        A list of strings with one string per line from the target filename.
    """

    with open(filename, "r") as file:
        return file.readlines()

def preprocess_text_with_line_numbers(filename):
    """
    Returns a list of dictionaries of abstract line data.

    Args:
        filename: Reads it's contents and sorts through each line,
                  extracting things like target label, the text of the sentence,
                  how many sentences are in the current abstract and what sentence
                  number the target line is.

    """
    input_lines = get_lines(filename)   # get all lines from filename
    abstract_lines = ""                 # create an empty abstract
    abstract_samples = []               # create an empty list of abstracts

    # Loop through each line in the target file
    for line in input_lines:
        if line.startswith("###"):  # check to see if the line is an ID line
            abstract_id = line
            abstract_lines = ""     # reset the abstract string if the line is an ID line
        elif line.isspace():        # check to see if line is a new line
            abstract_line_split = abstract_lines.splitlines()   # split abstract into separate lines

            # Iterate through each line in a single abstract and count them at the same time
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}                                          # create an empty dictionary for each line
                target_text_split = abstract_line.split("\t")           # split target label from text
                line_data["target"] = target_text_split[0]              # get the target label from text
                line_data["text"] = target_text_split[1].lower()        # get target text and lower it
                line_data["line_number"] = abstract_line_number         # what number line does the line appear in the abstract
                line_data["total_lines"] = len(abstract_line_split) - 1 # how many total lines are there in the target abstract? (start from 0)
                abstract_samples.append(line_data)                      # add line data to abstract samples list

        else:   # if the above conditions aren't fulfilled, the line contains a labelled sentence
            abstract_lines += line

    return abstract_samples

def get_abstract_lines(text):
    nlp = English() # setup English sentence parser
    nlp.add_pipe("sentencizer") # add sentence splitting pipeline object to sentence parser
    doc = nlp(text) # create "doc" of parsed sequences, change index for a different abstract
    abstract_lines = [str(sent) for sent in list(doc.sents)] # return detected sentences from doc in string type (not spaCy token type)
    return abstract_lines

def get_sample_lines(abstract_lines):
    total_lines_in_sample = len(abstract_lines)

    # Go through each line in abstract and create a list of dictionaries containing features for each line
    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)

    return sample_lines

def load_and_predict(text):

    abstract_lines = get_abstract_lines(text)
    sample_lines = get_sample_lines(abstract_lines)

    # Get all line_number values from sample abstract
    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]

    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15) 

    # Get all total_lines values from sample abstract
    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]

    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)

    # Split abstract lines into characters
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]

    loaded_model = tf.keras.models.load_model(current_dir + "/saved_models/model_5")

    test_abstract_pred_probs = loaded_model.predict(x=(test_abstract_line_numbers_one_hot,
                                                    test_abstract_total_lines_one_hot,
                                                    tf.constant(abstract_lines),
                                                    tf.constant(abstract_chars)))

    test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)

    data_dir = current_dir + "/pubmed-rct/PubMed_20k_RCT/"

    train_samples = preprocess_text_with_line_numbers(data_dir + "train.txt")

    train_df = pd.DataFrame(train_samples)

    # Extract labels ("target" columns) and encode them into integers
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())

    test_abstract_pred_classes = [label_encoder.classes_[i] for i in test_abstract_preds]

    results = []

    for i, line in enumerate(abstract_lines):
        results.append(f"{test_abstract_pred_classes[i]}: {line}")

    return results

# Home page
@app.route('/')
def home():
	return render_template('index.html')

# Prediction page
@app.route('/prediction', methods = ['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['textbox']
        results = load_and_predict(text)
        # print('\n'.join(results))
        return render_template('prediction copy.html', original=text, predictions=results)