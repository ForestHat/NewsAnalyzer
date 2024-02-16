#!/usr/bin/python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sys import argv

input_result: list[str] = []
output_result: list[str] = []

def read_file(path: str) -> list[str]:
    lines: list[str] = []

    with open(path, "r") as file:
        for line in file:
            lines.append(line.strip().lower())

    return lines

def preparate_data_to_train(train_files: list[str], results: list[int]) -> None:
    for i in range(0, len(train_files)):
        file: list[str] = read_file(train_files[i])
        input_result.extend(file)

        for o in range(0, len(file)):
            output_result.append(results[i])

def get_predict(example: str):
    vectorizer: TfidfVectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    X = vectorizer.fit_transform(input_result)
    clf: SVC = SVC(probability=True)
    clf.fit(X, output_result)

    score = clf.predict(vectorizer.transform([example]))

    return score

def save_to_file(clf: SVC, vectorizer: TfidfVectorizer) -> None:
    print("Save to file")
    with open("clf.pickle", "wb") as file1:
        pickle.dump(clf, file1)

    with open("vectorizer.pickle", "wb") as file2:
        pickle.dump(vectorizer, file2)

def read_from_file() -> list[SVC, TfidfVectorizer]:
    result = []

    with open("clf.pickle", "rb") as file1:
        result.append(pickle.load(file1))

    with open("vectorizer.pickle", "rb") as file2:
        result.append(pickle.load(file2))

    return result

preparate_data_to_train(["train_data/technology.txt",
                         "train_data/science.txt",
                         "train_data/politics.txt",
                         "train_data/games.txt"], [0, 1, 2, 3])

if argv[1] is not None:
    predict = get_predict(argv[1])

    if predict == [0]:
        print("technology")
    elif predict == [1]:
        print("science")
    elif predict == [2]:
        print("politics")
    else:
        print("games")
else:
    print('Give the test data! Example: python3 main.py "Test example"')
