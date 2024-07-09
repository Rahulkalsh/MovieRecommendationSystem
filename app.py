from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import time
import requests
from bs4 import BeautifulSoup
from keras.models import load_model
import tensorflow as tf
# import subprocess
app = Flask(__name__)

# Load pre-trained emotion detection model
model = tf.keras.models.load_model('best_model.h5')

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_duration = 5  # in seconds

# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/search', methods=['POST'])
# def index():
#     return render_template('Search.html')

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    emotions = []

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            face_img = gray[y:y+h, x:x+w]

            face_img = cv2.resize(face_img, (48, 48))
            img = extract_features(face_img)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            emotions.append(prediction_label)

            txt = "Emotion: " + prediction_label
            cv2.putText(frame, txt, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('frame', frame)

        if time.time() - start_time >= video_duration:
            break

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    most_common_emotion = max(set(emotions), key=emotions.count)

    movie_data = fetch_movies_from_imdb(most_common_emotion)
    # print("Movie Data:", movie_data)  # Debugging print statement

    return render_template('result.html', emotion=most_common_emotion, movies=movie_data)

# @app.route('/search', methods=['GET', 'POST'])
# def search():
#     #Launch the Streamlit app using subprocess
#     subprocess.Popen(['streamlit', 'run', 'app1.py'])
#     return ''

def fetch_movies_from_imdb(emotion):
    genre_mapping = {
        'sad': 'drama',
        'disgust': 'musical',
        'angry': 'family',
        'neutral': 'thriller',
        'fear': 'horror',
        'happy': 'comedy',
        'surprise': 'film-noir'
    }

    genre = genre_mapping.get(emotion)

    if genre:
        url = f'http://www.imdb.com/search/title?genres={genre}&title_type=feature&sort=moviemeter, asc'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            movie_titles = soup.find_all('h3', class_='ipc-title__text')
            movie_ratings = soup.find_all('span', class_='ratingGroup--imdb-rating')
            movie_images = soup.find_all('img', class_='ipc-image')

            if len(movie_titles) == len(movie_images):
                movie_data = []
                for title, rating, image in zip(movie_titles, movie_ratings, movie_images):
                    title_text = title.text.strip()
                    if(rating):
                         rating_text = rating.text.strip()
                    else :
                        rating_text = 'N/A'     
                    image_url = image['src']
                    movie_data.append({'title': title_text, 'rating': rating_text, 'image_url': image_url})
                return movie_data
            else:
                print("Mismatch in the number of movie titles, ratings, and images.")
                print("Movie Titles:", len(movie_titles))
                print("Movie Ratings:", len(movie_ratings))
                print("Movie Images:", len(movie_images))
                return []
        else:
            print("Failed to fetch data from IMDb. Status code:", response.status_code)
            return []
    else:
        print("Invalid emotion specified.")
        return []

if __name__ == '__main__':
    app.run()
