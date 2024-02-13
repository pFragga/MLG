from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

def preprocess(file: str) -> str:
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # Specify paths for training and testing data
    train_folder = 'C:\\Users\\Taner\\Desktop\\texniti2\\Τεχνητη 2\\aclImdb_v1\\aclImdb\\train\\pos'
    test_folder = 'C:\\Users\\Taner\\Desktop\\texniti2\\Τεχνητη 2\\aclImdb_v1\\aclImdb\\test\\pos'

    # Collect file paths
    train_files = [os.path.join(train_folder, file) for file in os.listdir(train_folder) if file.endswith('.txt')]
    test_files = [os.path.join(test_folder, file) for file in os.listdir(test_folder) if file.endswith('.txt')]

    # Read and preprocess data
    train_data = [preprocess(file) for file in train_files]
    test_data = [preprocess(file) for file in test_files]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)

    # Labels for classification (assuming positive reviews)
    y_train = [1] * len(train_files)
    y_test = [1] * len(test_files)

    # Train Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Evaluate classifier on test data
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Prints for debugging
    print("Number of samples in the test set:", len(y_test))
    print("Number of unique predicted classes:", len(set(y_pred)))

if __name__ == "__main__":
    main()

