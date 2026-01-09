from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

# Load pickle files with pandas for compatibility
popular_df = pd.read_pickle('popular.pkl')
pt = pd.read_pickle('pt.pkl')
books = pd.read_pickle('books.pkl')
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))

# Load sentiment analysis results
sentiment_df = pd.read_csv('book_sentiment_analysis_results.csv')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    
    # Check if the book exists in the index
    matching_books = np.where(pt.index == user_input)[0]
    
    if len(matching_books) == 0:
        # Book not found, return error message
        return render_template('recommend.html', error=f"Book '{user_input}' not found. Please try another book title.")
    
    index = matching_books[0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        book_title = list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values)[0]
        
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        # Get sentiment analysis data for this book using partial matching
        sentiment_data = sentiment_df[sentiment_df['Title'].str.contains(book_title, case=False, na=False, regex=False)]
        if sentiment_data.empty:
            # Try reverse match: check if book_title contains any sentiment title
            for idx, row in sentiment_df.iterrows():
                if row['Title'].lower() in book_title.lower():
                    sentiment_data = sentiment_df.iloc[[idx]]
                    break
        
        if not sentiment_data.empty:
            positive_pct = sentiment_data.iloc[0]['positive_percentage']
            negative_pct = sentiment_data.iloc[0]['negative_percentage']
        else:
            positive_pct = 'N/A'
            negative_pct = 'N/A'
        
        item.append(positive_pct)
        item.append(negative_pct)

        data.append(item)

    print(data)

    return render_template('recommend.html',data=data)

if __name__ == '__main__':
    app.run(debug=True)