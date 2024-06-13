import pandas as pd
import re
import numpy as np
import itertools
import argparse
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_training_data(file_path):
    """
    Reads training data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the training data.
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna('empty', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error reading the file {file_path}: {e}")
        raise

def process_text(text, stop_words, ps):
    """
    Processes text by removing non-alphabetic characters, converting to lowercase,
    removing stopwords, and applying stemming.
    
    Args:
        text (str): The text to process.
        stop_words (set): Set of stopwords to remove.
        ps (PorterStemmer): PorterStemmer instance for stemming.
        
    Returns:
        str: The processed text.
    """
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

def stem_texts(data, n_jobs=-1):
    """
    Applies stemming to a series of texts in parallel.
    
    Args:
        data (pd.Series): Series containing the texts to process.
        n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.
        
    Returns:
        list: List of processed texts.
    """
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    texts = data['text']
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_text)(text, stop_words, ps) for text in tqdm(texts, desc="Stemming progress")
    )
    
    return results

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots a confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix.
        classes (list): List of class names.
        normalize (bool): Whether to normalize the confusion matrix.
        title (str): Title of the plot.
        cmap: Colormap for the plot.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logging.info("Normalized confusion matrix")
    else:
        logging.info('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]:.2f}',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main(file_path, output_file):
    """
    Main function to train the model.
    
    Args:
        file_path (str): Path to the CSV file containing training data.
        output_file (str): Path to save the trained model.
    """
    data = get_training_data(file_path)
    stem_data = stem_texts(data)
    
    tfidf_v = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
    x = tfidf_v.fit_transform(stem_data).toarray()
    y = data['label']
    
    with open('tfidf_vector.pickle', 'wb') as vec_file:
        pickle.dump(tfidf_v, vec_file)
    logging.info('Vector has been saved as tfidf_vector.pickle')
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
    
    logging.info('Training Passive Aggressive Classifier')
    pac = PassiveAggressiveClassifier(max_iter=50, random_state=0)
    pac.fit(x_train, y_train)
    
    logging.info('Making predictions and evaluating the model')
    y_pred = pac.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy:.3f}")
    
    cm = metrics.confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])
    
    with open(output_file, 'wb') as model_file:
        pickle.dump(pac, model_file)
    logging.info(f"Model has been saved as {output_file} with {accuracy * 100:.2f}% accuracy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file containing training data")
    parser.add_argument('file_path', type=str, help='Path to the CSV file containing training data.')
    parser.add_argument('--output_file', type=str, help='Name of the output model file.', default="Model.sav")
    args = parser.parse_args()
    
    main(args.file_path, args.output_file)
