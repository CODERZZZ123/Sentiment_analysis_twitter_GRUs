from flask import Flask, render_template, request
from dummy import *
from utils import *

app = Flask(__name__)

tweet_sentiment_analysis = TweetSentimentAnalysis()
tweet_sentiment_analysis.load_tweets()
min_occurrence = 0
tweet_sentiment_analysis.get_vocab(tweet_sentiment_analysis.train_x, min_occurrence)
tweet_sentiment_analysis.save_vocab()
length_vocab = len(tweet_sentiment_analysis.Vocab)
sentiment_analysis_model = SentimentAnalysisModel(vocab_size=length_vocab)
sentiment_analysis_model.load_model()

def predict_object(sentence):
    try:
        pred , sentiment = sentiment_analysis_model.predict(sentence,tweet_sentiment_analysis.Vocab)
        return pred , sentiment

    except Exception as e:
        # Handle the exception and display a warning message
        warning_message = f"An error occurred: {str(e)}"
        return None, warning_message

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form['sentence']
        sentence = str(sentence)
        tmp_pred, tmp_sentiment = predict_object(sentence)
        print(tmp_sentiment)
        return render_template('index.html', sentence=sentence, prediction=tmp_sentiment)

if __name__ == '__main__':
    app.run(debug=True, port=5002)