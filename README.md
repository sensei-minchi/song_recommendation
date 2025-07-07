# song_recommendation
Song recommendation system created by using content-based filtering algorithm which recommends items using features of the items with what the user has liked in the past.

The dataset has been taken from kaggle https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset

##Content based filtering
A recommender algorithm which tries to recommend items to user using features of the items with what user has liked in the past. To create features of songs, Tfidf Vectorizer has been used which converts text into numerical form based on how important each word is. After extracting features of songs, a neural network has been created which would try to predict whether user would like the item or not. Before the model starts giving out some relevant recommendations, we would need to get some data about user and what the user would like (cold start)

