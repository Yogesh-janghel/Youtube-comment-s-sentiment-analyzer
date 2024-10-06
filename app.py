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

app = Flask(__name__)

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
    
    # Preprocess the comment (tokenization and padding)
    new_sequence = tokenizer.texts_to_sequences([comment])
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
#Route for processing YouTube URL submission

# @app.route('/submit_url', methods=['POST'])
# def submit_url():
    youtube_url = request.form.get('youtube_url') 
    if not youtube_url:
        return "No URL provided", 400
    
    api_service_name = "youtube"
    api_version = "v3"
        
    # Build the YouTube API client
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY
    )

    video_id = extract_video_id(youtube_url)
    comments = []
    next_page_token = None
    max_comments = 100

    video_request = youtube.videos().list(
        part="snippet",
        id=video_id
    )
    video_response = video_request.execute()
    video_title = video_response['items'][0]['snippet']['title']

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
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['updatedAt'],
                comment['likeCount'],
                comment['textDisplay'],
                len(comment['textDisplay'])  # Add comment length for later analysis
            ])

            if len(comments) >= max_comments:
                break

        # Break the outer loop if max_comments reached
        if len(comments) >= max_comments:
            break

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text', 'comment_length']) 

    # Save the first graph (Distribution of Likes) as an image
    plt.figure(figsize=(10, 6))
    df['like_count'].hist(bins=30, color='blue', edgecolor='black')
    plt.title('Distribution of Comment Likes')
    plt.xlabel('Like Count')
    plt.ylabel('Number of Comments')
    plt.xlim(0, 2000)
    plt.savefig('static/images/like_distribution.png')  # Save as image
    plt.close()

    # Save the second graph (Correlation) as an image
    plt.figure(figsize=(10, 6))
    plt.scatter(df['comment_length'], df['like_count'], alpha=0.5)
    plt.title('Correlation Between Comment Length and Likes')
    plt.xlabel('Comment Length')
    plt.ylabel('Like Count')
    plt.savefig('static/images/comment_length_vs_likes.png')  # Save as image
    plt.close()

    # Add sentiment analysis for comments
    sentiments = []
    for comment in df['text']:
        new_sequence = tokenizer.texts_to_sequences([comment])
        new_padded = pad_sequences(new_sequence, maxlen=max_seq_length)
        lstm_pred = model_lstm.predict(new_padded)
        sentiment_score = np.argmax(lstm_pred)
        sentiments.append(sentiment_score)
    
    df['sentiment'] = sentiments

     # Filter top 5 most liked positive and negative comments
    top_positive_comments = df[df['sentiment'] == 2].nlargest(5, 'like_count')
    top_negative_comments = df[df['sentiment'] == 0].nlargest(5, 'like_count')

    # Prepare data for rendering
    top_positive_comments_list = top_positive_comments[['text', 'like_count']].to_dict(orient='records')
    top_negative_comments_list = top_negative_comments[['text', 'like_count']].to_dict(orient='records')


    # Calculate overall sentiment
    sentiment_counts = df['sentiment'].value_counts()
    overall_sentiment = sentiment_counts.idxmax()

    if overall_sentiment == 0:
        sentiment = "Negative"
    elif overall_sentiment == 1:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    # Create a pie chart for sentiment distribution
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts.values, labels=['Negative', 'Neutral', 'Positive'], autopct='%1.1f%%')
    plt.title('Sentiment Distribution of Comments')
    plt.savefig('static/images/sentiment_distribution.png')
    plt.close()

    # Save the third graph (Comment Activity Over Time) as an image
    df['published_at'] = pd.to_datetime(df['published_at'])
    plt.figure(figsize=(12, 6))
    df.set_index('published_at').resample('D').size().plot(title='Comment Activity Over Time')
    plt.ylabel('Number of Comments')
    plt.savefig('static/images/comment_activity_over_time.png')  # Save as image
    plt.close()

    # Save the fourth graph (Top 10 Most Active Authors) as an image
    top_authors = df['author'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    top_authors.plot(kind='bar', color='green')
    plt.title('Top 10 Most Active Authors')
    plt.xlabel('Author')
    plt.ylabel('Number of Comments')
    plt.savefig('static/images/top_authors.png')  # Save as image
    plt.close()

    # Save the fifth graph (Distribution of Comment Lengths) as an image
    df['comment_length'] = df['text'].apply(len)
    plt.figure(figsize=(10, 6))
    df['comment_length'].hist(bins=30, color='orange', edgecolor='black')
    plt.title('Distribution of Comment Lengths')
    plt.xlabel('Comment Length')
    plt.ylabel('Number of Comments')
    plt.xlim(0, 2500)
    plt.savefig('static/images/comment_length_distribution.png')  # Save as image
    plt.close()

    # Save the sixth graph (Comment Activity by Hour) as an image
    df['hour'] = df['published_at'].dt.hour
    comment_hours = df.groupby('hour').size().reset_index(name='count')
    comment_hours_pivot = comment_hours.pivot_table(values='count', index='hour', aggfunc='sum')
    plt.figure(figsize=(12, 6))
    sns.heatmap(comment_hours_pivot, annot=True, fmt='d', cmap='Blues')
    plt.title('Comment Activity by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Comments')
    plt.savefig('static/images/comment_activity_by_hour.png')  # Save as image
    plt.close()

    # Save the seventh graph (Word Cloud) as an image
    text = ' '.join(df['text'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Comments')
    plt.savefig('static/images/wordcloud.png')  # Save as image
    plt.close()

    # last_video_id=video_id
    # last_title=video_title
    # sentiment_last=sentiment

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
                           top_positive_comments=top_positive_comments_list,
                           top_negative_comments=top_negative_comments_list
                        )

def fetch_video_details(youtube, video_id):
    video_request = youtube.videos().list(
        part="snippet",
        id=video_id
    )
    video_response = video_request.execute()
    video_title = video_response['items'][0]['snippet']['title']
    return video_title

# Function to fetch comments
def fetch_comments(youtube, video_id, max_comments=100):
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
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['updatedAt'],
                comment['likeCount'],
                comment['textDisplay'],
                len(comment['textDisplay'])  # Add comment length for later analysis
            ])
            if len(comments) >= max_comments:
                break

        if len(comments) >= max_comments or not response.get('nextPageToken'):
            break
        next_page_token = response.get('nextPageToken')

    df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text', 'comment_length'])
    return df

# Function to create and save various plots
def create_plots(df):
    # Like distribution
    plt.figure(figsize=(10, 6))
    df['like_count'].hist(bins=30, color='blue', edgecolor='black')
    plt.title('Distribution of Comment Likes')
    plt.xlabel('Like Count')
    plt.ylabel('Number of Comments')
    plt.xlim(0, 2000)
    plt.savefig('static/images/like_distribution.png')
    plt.close()

    # Comment length vs likes correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(df['comment_length'], df['like_count'], alpha=0.5)
    plt.title('Correlation Between Comment Length and Likes')
    plt.xlabel('Comment Length')
    plt.ylabel('Like Count')
    plt.savefig('static/images/comment_length_vs_likes.png')
    plt.close()

    # Comment activity over time
    df['published_at'] = pd.to_datetime(df['published_at'])
    plt.figure(figsize=(12, 6))
    df.set_index('published_at').resample('D').size().plot(title='Comment Activity Over Time')
    plt.ylabel('Number of Comments')
    plt.savefig('static/images/comment_activity_over_time.png')
    plt.close()

    # Top 10 most active authors
    top_authors = df['author'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    top_authors.plot(kind='bar', color='green')
    plt.title('Top 10 Most Active Authors')
    plt.xlabel('Author')
    plt.ylabel('Number of Comments')
    plt.savefig('static/images/top_authors.png')
    plt.close()

    # Comment length distribution
    plt.figure(figsize=(10, 6))
    df['comment_length'].hist(bins=30, color='orange', edgecolor='black')
    plt.title('Distribution of Comment Lengths')
    plt.xlabel('Comment Length')
    plt.ylabel('Number of Comments')
    plt.xlim(0, 2500)
    plt.savefig('static/images/comment_length_distribution.png')
    plt.close()

    # Comment activity by hour
    df['hour'] = df['published_at'].dt.hour
    comment_hours = df.groupby('hour').size().reset_index(name='count')
    comment_hours_pivot = comment_hours.pivot_table(values='count', index='hour', aggfunc='sum')
    plt.figure(figsize=(12, 6))
    sns.heatmap(comment_hours_pivot, annot=True, fmt='d', cmap='Blues')
    plt.title('Comment Activity by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Comments')
    plt.savefig('static/images/comment_activity_by_hour.png')
    plt.close()

    # Word cloud
    text = ' '.join(df['text'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Comments')
    plt.savefig('static/images/wordcloud.png')
    plt.close()

# Function for sentiment analysis
def perform_sentiment_analysis(df, tokenizer, model_lstm, max_seq_length):
    sentiments = []
    for comment in df['text']:
        new_sequence = tokenizer.texts_to_sequences([comment])
        new_padded = pad_sequences(new_sequence, maxlen=max_seq_length)
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
    if not youtube_url:
        return "No URL provided", 400

    api_service_name = "youtube"
    api_version = "v3"
        
    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

    video_id = extract_video_id(youtube_url)
    video_title = fetch_video_details(youtube, video_id)
    df = fetch_comments(youtube, video_id, max_comments=100)

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



if __name__ == '__main__':
    app.run(debug=True)

