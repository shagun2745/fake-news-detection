print("STARTING SCRIPT...")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

print("Creating dummy dataset...")

# small manual dataset (no CSV)
texts = [
    "This is fake news",
    "Breaking: government announces new policy",
    "Click here to win money now",
    "Official report confirms the results",
    "Shocking secret revealed!!!",
    "The study shows positive results"
]

labels = [0, 1, 0, 1, 0, 1]

print("Vectorizing...")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

print("Training model...")

model = LogisticRegression()
model.fit(X, labels)

print("Saving model...")

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("DONE ✅")