import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Step 1: Read the data from the provided files

def read_data(text_file, labels_file):
    texts = []
    labels = []
    with open(text_file, 'r', encoding='utf-8') as text_f, open(labels_file, 'r', encoding='utf-8') as labels_f:
        for text_line, label_line in zip(text_f, labels_f):
            text_data = json.loads(text_line)
            label_data = json.loads(label_line)
            texts.append(text_data['text'])
            labels.append(label_data['generated'])
    return texts, labels

train_texts, train_labels = read_data('train_text.jsonl', 'train_labels.jsonl')
test_texts, _ = read_data('test_text.jsonl', 'test_labels.jsonl')

# Step 2: Preprocess the text data and train the model

vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_texts)
y_train = train_labels

model = LogisticRegression()
model.fit(X_train, y_train)

# Step 3: Make predictions on the test data

X_test = vectorizer.transform(test_texts)
predictions = model.predict(X_test)

# Step 4: Save predictions into predictions.jsonl file

with open('predictions.jsonl', 'w', encoding='utf-8') as f:
    for idx, pred in enumerate(predictions):
        prediction_data = {'id': idx, 'generated': int(pred)}
        f.write(json.dumps(prediction_data) + '\n')

# Step 5: Calculate F1 score
# Since we don't have ground truth labels for the test set, we cannot calculate the F1 score here.

# However, you can calculate it during the evaluation step when submitting to TIRA.
