import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


df = pd.read_csv('spam.csv', encoding='ISO-8859-1')[['v1', 'v2']]
df.columns = ['label', 'message']


df['label'] = df['label'].map({'spam': 1, 'ham': 0})


ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


df['transformed_message'] = df['message'].apply(transform_text)

spam = df[df['label'] == 1]
ham = df[df['label'] == 0]


ham = ham.sample(n=len(spam), random_state=42)


balanced_df = pd.concat([spam, ham])

num_spam = balanced_df[balanced_df['label'] == 1].shape[0]
num_ham = balanced_df[balanced_df['label'] == 0].shape[0]
print(f"Number of spam messages: {num_spam}")
print(f"Number of ham messages: {num_ham}")


X_train, X_test, y_train, y_test = train_test_split(balanced_df['transformed_message'], balanced_df['label'], test_size=0.2, random_state=42)


tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


def predict_spam(sms_text):

    transformed_sms = transform_text(sms_text)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    return "Spam" if result == 1 else "Not Spam"


while True:
    input_sms = input("Enter the message to check if it's spam or not (or type 'exit' to quit): ")

    if input_sms.lower() == 'exit':
        print("Exiting the program.")
        break

    prediction = predict_spam(input_sms)
    print(f"Prediction: {prediction}")