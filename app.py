import os
from flask import Flask, request, render_template, send_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import googleapiclient.discovery
import googleapiclient.errors
from urllib.parse import urlparse, parse_qs
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from urllib.parse import urlparse, parse_qs
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect
from googletrans import Translator
import plotly.express as px

app = Flask(__name__)

translator = Translator()

sentiment_last="Positive"
last_video_id="DLET_u31M4M"
last_title="IT ENDS WITH US - Official Trailer (HD)"

DEVELOPER_KEY = "AIzaSyB_dxNMTBGn3iPER2YF_HItaV-WS_I0WVE"
if not os.path.exists('static/images'):
    os.makedirs('static/images')

# Load your LSTM model
model_lstm = load_model('lstm_sentiment_model.h5')

# Load tokenizer (make sure you have a tokenizer saved in your project)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Parameters for padding
max_seq_length = 100  # Update with your actual sequence length used in training

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for the About page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for processing comment submission and predicting sentiment
@app.route('/submit_comment', methods=['POST'])
def submit_comment():
    comment = request.form.get('comment')
    
    # Clean and preprocess the comment
    cleaned_comment = clean_and_preprocess_comments(comment)
    
    # Tokenize and pad the cleaned comment
    new_sequence = tokenizer.texts_to_sequences([cleaned_comment])
    new_padded = pad_sequences(new_sequence, maxlen=max_seq_length)
    
    # Predict sentiment using LSTM model
    lstm_pred = model_lstm.predict(new_padded)
    
    # Get the predicted sentiment score and sentiment label (0, 1, 2)
    sentiment_score = np.argmax(lstm_pred)
    
    if sentiment_score == 0:
        sentiment_label = "Negative"
    elif sentiment_score == 1:
        sentiment_label = "Neutral"
    else:
        sentiment_label = "Positive"
    
    # Pass sentiment prediction and score to comment.html
    return render_template('comment.html', sentiment_label=sentiment_label, sentiment_score=lstm_pred[0][sentiment_score])

def fetch_video_details(youtube, video_id):
    video_request = youtube.videos().list(
        part="snippet",
        id=video_id
    )
    video_response = video_request.execute()
    video_title = video_response['items'][0]['snippet']['title']
    return video_title

def translate_to_english(comment):
    try:
        translation = translator.translate(comment, src='hi', dest='en')
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
    return comment

def detect_hinglish(comment):
    try:
        lang = detect(comment)
        # Simple heuristic to identify Hinglish
        if lang == 'en' and any(word in comment for word in ['hai', 'kya', 'nahi', 'kaise']):
            return True
        return False
    except:
        return False
    
# Function to fetch comments
def fetch_comments(youtube, video_id, max_comments=100, max_comment_length=150):
    comments = []
    next_page_token = None
    while True:
        yt_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = yt_request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comment_text = comment['textDisplay']
            if len(comment_text) <= max_comment_length:
                comments.append([
                    comment['authorDisplayName'],
                    comment['publishedAt'],
                    comment['updatedAt'],
                    comment['likeCount'],
                    comment_text,
                    len(comment_text)  # Add comment length for later analysis
                ])

        if len(comments) >= max_comments or not response.get('nextPageToken'):
            break
        next_page_token = response.get('nextPageToken')

    df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text', 'comment_length'])
    return df

# Function to create and save various plots
def create_plots(df):
    # Like distribution
    fig = px.histogram(df, x='like_count', nbins=30, title='Distribution of Comment Likes')
    fig.update_layout(xaxis_title='Like Count', yaxis_title='Number of Comments')
    fig.write_image('static/images/like_distribution.png')

    # Comment length vs likes correlation
    fig = px.scatter(df, x='comment_length', y='like_count', opacity=0.5, title='Correlation Between Comment Length and Likes')
    fig.update_layout(xaxis_title='Comment Length', yaxis_title='Like Count')
    fig.write_image('static/images/comment_length_vs_likes.png')

    # Comment activity over time
    df['published_at'] = pd.to_datetime(df['published_at'])
    comment_activity = df.set_index('published_at').resample('D').size().reset_index(name='count')
    fig = px.line(comment_activity, x='published_at', y='count', title='Comment Activity Over Time')
    fig.update_layout(xaxis_title='Date', yaxis_title='Number of Comments')
    fig.write_image('static/images/comment_activity_over_time.png')

    # Top 10 most active authors
    top_authors = df['author'].value_counts().head(10).reset_index()
    top_authors.columns = ['author', 'count']
    fig = px.bar(top_authors, x='author', y='count', title='Top 10 Most Active Authors')
    fig.update_layout(xaxis_title='Author', yaxis_title='Number of Comments')
    fig.write_image('static/images/top_authors.png')

    # Comment length distribution
    fig = px.histogram(df, x='comment_length', nbins=30, title='Distribution of Comment Lengths')
    fig.update_layout(xaxis_title='Comment Length', yaxis_title='Number of Comments')
    fig.write_image('static/images/comment_length_distribution.png')

    # Comment activity by hour
    df['hour'] = df['published_at'].dt.hour
    comment_hours = df.groupby('hour').size().reset_index(name='count')
    fig = px.density_heatmap(comment_hours, x='hour', y='count', title='Comment Activity by Hour')
    fig.update_layout(xaxis_title='Hour of the Day', yaxis_title='Number of Comments')
    fig.write_image('static/images/comment_activity_by_hour.png')

    # Word cloud
    text = ' '.join(df['text'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Comments')
    plt.savefig('static/images/wordcloud.png')
    plt.close()


# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean and preprocess comments
def clean_and_preprocess_comments(comment):

    if detect_hinglish(comment):
           comment = translate_to_english(comment)
    # Convert to lowercase
    comment = comment.lower()
    # Remove URLs
    comment = re.sub(r'http\S+|www\S+|https\S+', '', comment, flags=re.MULTILINE)
    # Remove mentions
    comment = re.sub(r'@\w+', '', comment)
    # Remove markdown
    comment = re.sub(r'\[.*?\]\(.*?\)', '', comment)
    # Remove punctuation
    comment = re.sub(r'[^\w\s]', '', comment)
    # Tokenize
    tokens = word_tokenize(comment)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function for sentiment analysis
def perform_sentiment_analysis(df, tokenizer, model_lstm, max_seq_length):
    # Clean and preprocess comments
    df['cleaned_text'] = df['text'].apply(clean_and_preprocess_comments)
    
    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
    X = vectorizer.fit_transform(df['cleaned_text']).toarray()
    
    sentiments = []
    for i in range(len(df)):
        # Tokenize and pad the cleaned comment
        new_sequence = tokenizer.texts_to_sequences([df['cleaned_text'].iloc[i]])
        new_padded = pad_sequences(new_sequence, maxlen=max_seq_length)
        # Predict sentiment
        lstm_pred = model_lstm.predict(new_padded)
        sentiment_score = np.argmax(lstm_pred)
        sentiments.append(sentiment_score)
    df['sentiment'] = sentiments
    return df

# Function to calculate overall sentiment
def calculate_overall_sentiment(df):
    sentiment_counts = df['sentiment'].value_counts()
    overall_sentiment = sentiment_counts.idxmax()

    if overall_sentiment == 0:
        sentiment = "Negative"
    elif overall_sentiment == 1:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"
    
    return sentiment, sentiment_counts

# Function to prepare top comments based on sentiment
def prepare_top_comments(df):
    top_positive_comments = df[df['sentiment'] == 2].nlargest(5, 'like_count')
    top_negative_comments = df[df['sentiment'] == 0].nlargest(5, 'like_count')

    top_positive_comments_list = top_positive_comments[['text', 'like_count']].to_dict(orient='records')
    top_negative_comments_list = top_negative_comments[['text', 'like_count']].to_dict(orient='records')

    return top_positive_comments_list, top_negative_comments_list

# Route handler
@app.route('/submit_url', methods=['POST'])
def submit_url():
    youtube_url = request.form.get('youtube_url')
    num_comments = int(request.form.get('num_comments', 100))
    max_comment_length = int(request.form.get('max_comment_length', 150))

    if not youtube_url:
        return "No URL provided", 400

    api_service_name = "youtube"
    api_version = "v3"
        
    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

    video_id = extract_video_id(youtube_url)
    video_title = fetch_video_details(youtube, video_id)
    df = fetch_comments(youtube, video_id, max_comments=num_comments, max_comment_length=max_comment_length)

    create_plots(df)

    # Perform sentiment analysis
    df = perform_sentiment_analysis(df, tokenizer, model_lstm, max_seq_length)
    
    # Calculate overall sentiment
    sentiment, sentiment_counts = calculate_overall_sentiment(df)

    # Prepare top positive and negative comments
    top_positive_comments, top_negative_comments = prepare_top_comments(df)

    # Create a pie chart for sentiment distribution
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts.values, labels=['Negative', 'Neutral', 'Positive'], autopct='%1.1f%%')
    plt.title('Sentiment Distribution of Comments')
    plt.savefig('static/images/sentiment_distribution.png')
    plt.close()

    return render_template('youtube.html', sentiment=sentiment, 
                           like_dist_image='/static/images/like_distribution.png',
                           comment_corr_image='/static/images/comment_length_vs_likes.png',
                           sentiment_dist_image='/static/images/sentiment_distribution.png',
                           comment_activity_image='/static/images/comment_activity_over_time.png',
                           top_authors_image='/static/images/top_authors.png',
                           comment_length_dist_image='/static/images/comment_length_distribution.png',
                           comment_activity_by_hour_image='/static/images/comment_activity_by_hour.png',
                           wordcloud_image='/static/images/wordcloud.png',
                           video_title=video_title,
                           video_id=video_id,
                           top_positive_comments=top_positive_comments,
                           top_negative_comments=top_negative_comments)
                           
def extract_video_id(url):
    # Parse the URL into components
    parsed_url = urlparse(url)
    
    # Handle URLs with query parameters (e.g., https://www.youtube.com/watch?v=VIDEO_ID)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
    
    # Handle shortened URLs (e.g., https://youtu.be/VIDEO_ID)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]  # Remove the leading '/'
    
    return None

@app.route('/last_fetched', methods=['POST'])
def last_fetch_fucn():
    return render_template('last_fetch.html', sentiment=sentiment_last, 
                           like_dist_image='/static/images/like_distribution.png',
                           comment_corr_image='/static/images/comment_length_vs_likes.png',
                           sentiment_dist_image='/static/images/sentiment_distribution.png',
                           comment_activity_image='/static/images/comment_activity_over_time.png',
                           top_authors_image='/static/images/top_authors.png',
                           comment_length_dist_image='/static/images/comment_length_distribution.png',
                           comment_activity_by_hour_image='/static/images/comment_activity_by_hour.png',
                           wordcloud_image='/static/images/wordcloud.png',
                           video_title=last_title,
                           video_id=last_video_id)

@app.route('/how_it_works')
def how_it_works():
    return render_template('how_it_works.html')

if __name__ == '__main__':
    app.run(debug=True)

