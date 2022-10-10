import sys
# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet') # download for lemmatization
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    ''' This function load the data from the database. It returns messages column, categories dataframe, and categories names '''

    # load data from database
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse', engine)
    # Set the message column as X
    X = df.message
    # Set the other 36 categories as Y
    Y = df[df.columns[4:]]
    # Get category names
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    ''' This function normalize, tokenize, stem, and lemmatize the text. It returns the processed text '''
    
    # Normalize the text (case normalization & punctuation removal)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize the text by split text into words using NLTK
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Stemming by using the `PorterStemmer()` to perform stemming on the words processed above.
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    # Lemmatization using the 'WorldNetLemmatizer()' to perform lemmatization on the `words` variable.
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]

    return lemmed

def build_model():
    ''' This function builds a pipeline and perform grid search to find the best parameters. It returns a model '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        # Since we want to classify 36 category, the 'MultiOutputClassifier' is used with the 'RandomForestClassifier', 
        #   which does not support multi-target classification. 
        # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
        
    # References:
    # - https://towardsdatascience.com/random-forest-hyperparameters-and-how-to-fine-tune-them-17aee785ee0d
    # - https://medium.com/@benfenison/gridsearching-a-random-forest-classifier-fc225609699c
    parameters = {
        'clf__estimator__n_estimators': [5, 10]} # The Nº of Decision Trees in the forest
        # Commented the below due to the running time...
    #     'clf__estimator__n_estimators': [50, 100, 200], # The Nº of Decision Trees in the forest 
    #     'clf__estimator__criterion': ['gini','entropy']} # The criteria with which to split on each node (Gini or Entropy for a classification task) 

    # Perform grid search
    # Added n_jobs=-1 to use multiple cores of the processor which will speed up the process. Reference: https://stackoverflow.com/questions/17455302/gridsearchcv-extremely-slow-on-small-dataset-in-scikit-learn
    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' This function evaluates a model and displays the accuracy, precision, and recall of each class'''
    predicted = model.predict(X_test)
    
    for i in range(len(Y_test.columns)):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.iloc[:,i], predicted[:,i]))
    accuracy = (predicted == Y_test).mean()
    print("Accuracy:", accuracy)

def save_model(model, model_filepath):
    ''' This function exports the model as a Pickle file'''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()