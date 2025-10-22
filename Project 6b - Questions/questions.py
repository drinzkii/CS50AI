import os
import sys
import string
import math
import nltk
from collections import defaultdict

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

nltk.download("stopwords")
nltk.download("punkt")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    data = {
        filename: open(os.path.join(directory, filename), "r", encoding="utf-8").read()
        for filename in os.listdir(directory)
        if filename.endswith(".txt")
    }
    return data


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Tokenize the document, convert to lowercase, and remove punctuation
    tokens = [
        word.lower()
        for word in word_tokenize(document)
        if word not in string.punctuation
    ]
    # Filter out stopwords
    filtered_tokens = [
        word for word in tokens if word not in stopwords.words("english")
    ]
    return filtered_tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = defaultdict(float)
    total_docs = len(documents)
    word_counts = defaultdict(int)
    for doc, words in documents.items():
        seen_words = set(words)
        for word in seen_words:
            word_counts[word] += 1
    for word, count in word_counts.items():
        idfs[word] = math.log(total_docs / count)
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs = {
        file: sum(words.count(word) * idfs[word] for word in query)
        for file, words in files.items()
    }
    sorted_files = sorted(tf_idfs.keys(), key=lambda file: tf_idfs[file], reverse=True)
    return sorted_files[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    scores = defaultdict(float)
    for sentence, words in sentences.items():
        matching_word_count = sum(1 for word in words if word in query)
        query_term_density = matching_word_count / len(words)
        scores[sentence] = (
            sum(idfs[word] for word in query if word in words) + query_term_density
        )

    return sorted(scores.keys(), key=lambda s: scores[s], reverse=True)[:n]


if __name__ == "__main__":
    main()
