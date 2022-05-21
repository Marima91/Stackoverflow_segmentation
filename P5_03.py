from flask import Flask
from flask_restful import Api, Resource
import pandas as pd
import joblib
import spacy
import en_core_web_sm
import preprocessing as ppc


app = Flask(__name__)
api = Api(app)


# Load pre-trained models
vectorizer = joblib.load("tfidf_vectorizer.pkl", 'r')
multilabel_binarizer = joblib.load("multilabel_binarizer.pkl", 'r')
model = joblib.load("logit_nlp_model.pkl", 'r')

class Autotag(Resource):
    def get(self, question):
        """
       This examples uses FlaskRESTful Resource for Stackoverflow auto-tagging questions
       To test, copy and paste a non-cleaned question (even with HTML tags or code) and execute the model.
       ---
       parameters:
         - in: path
           name: question
           type: string
           required: true
       
        """
        # Clean the question sent
        nlp = en_core_web_sm.load(exclude=['ner', 'parser'])
        pos_list = ["NOUN","PROPN"]
        rawtext = question
        cleaned_question = ppc.text_cleaner(rawtext, nlp, pos_list, "english")
        
        # Apply saved trained TfidfVectorizer
        X_tfidf = vectorizer.transform([cleaned_question])
        
        # Perform prediction
        predict = model.predict(X_tfidf)
        
        # Inverse multilabel binarizer
        tags_predict = multilabel_binarizer.inverse_transform(predict)
            
        # Results
        results = {}
        results['Predicted_Tags'] = tags_predict
        return results


api.add_resource(Autotag, '/autotag/<question>')

if __name__ == "__main__":
	app.run()