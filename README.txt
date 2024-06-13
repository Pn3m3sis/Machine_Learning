Name: Model_Creator.py

Description: Creates a supervised machine learning model using scikit-learn and NLTK (natural language tool kit.) It takes dataset containing labeled articles and traines the model with Term Frequency Inversed document frequency (TFIDF) and PassiveAgressive Classifier. The script can take any dataset as long as the column that contains articles is named "text" and the column that is refers to reliable or unreliable articles are named "label" and reliable = 0 and unreliable = 1. NLTK is utilized for word stemming for feature extraction and removing stopwords.

Prerequesits:	
Python
Scikit-learn
nltk
tqdm
matplotlib
pandas
numpy
joblib

Output:	
Model = Model.sav (can be choosen by using arg: --output_file "filename.sav")
vector = tfidf_vector.pickle

Use:
python Model_Creator.py "filepath of training data" 
(optional) --output_file "filename.sav"

Testfile:
Name: model_test.py

Description: Model test is a short program that contains one unreliable and one reliable article. It prompts the user to select one of the two articles then prints it out to screen along with the title. It will then perform language processing on the article and fit it into the same vector as used by the model, (tfidf_vector.pickle) and then use the predict function to get a boolean value indicating that the article is either reliable or unreliable and print it to screen.

Use:
python model_test.py "filename.sav"