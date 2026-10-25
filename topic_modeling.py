import re
import webbrowser
from pathlib import Path

import pandas as pd
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF

import pyLDAvis
import pyLDAvis.gensim_models

import matplotlib.pyplot as plt
from wordcloud import WordCloud

BASE_DIR = Path(__file__).parent
file_path = BASE_DIR / 'voted-kaggle-dataset.csv'
df = pd.read_csv(file_path)

print("Dataset preview:")
print(df.head())

print("\nDataset columns:")
print(df.columns)

print("\nDataset information:")
df.info()
df = df.dropna(subset=['Description']).copy()
df['Description'] = df['Description'].astype(str).str.strip() # Remove descriptions containing only whitespace
df = df[df['Description'] != ''].copy() # Remove any remaining whitespace-only descriptions
df = df[
    df['Description'].str.contains(r'\S', regex=True)
].copy()
df.reset_index(drop=True, inplace=True)

print("\nNumber of cleaned descriptions:", len(df))

#Remove stop words and tokenize text
# Combine sklearn's English stop words with custom stop words
custom_stop_words = {
    'using', 'use', 'used', 'also', 'one', 'two',
    'new', 'get', 'make', 'many', 'much', 'like'
}

stop_words = set(ENGLISH_STOP_WORDS).union(custom_stop_words)


def preprocess_text(text):
    # Lowercase and tokenize; discard very short tokens
    tokens = simple_preprocess(
        str(text),
        deacc=True,
        min_len=2
    )

    # Remove stop words
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return tokens


# Tokenized descriptions for LDA
tokenized_data = [
    preprocess_text(text)
    for text in df['Description']
]

# Remove documents that contain no tokens after cleaning
valid_indices = [
    i for i, tokens in enumerate(tokenized_data)
    if len(tokens) > 0
]

df = df.iloc[valid_indices].reset_index(drop=True)

tokenized_data = [
    tokenized_data[i] for i in valid_indices
]

if not tokenized_data:
    raise ValueError(
        "No usable text remains after cleaning. "
        "Check your descriptions and stop-word list."
    )

cleaned_text = [
    ' '.join(tokens) for tokens in tokenized_data
]

tfidf_vectorizer = TfidfVectorizer(
    stop_words=list(stop_words),
    max_features=5000
)

tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_text)

print("\nTF-IDF matrix shape:", tfidf_matrix.shape)
dictionary = corpora.Dictionary(tokenized_data)

# Optional: remove extremely rare and overly common terms
dictionary.filter_extremes(
    no_below=2,
    no_above=0.90
)

corpus = [
    dictionary.doc2bow(tokens)
    for tokens in tokenized_data
]

# Exclude documents that have no words remaining
valid_docs = [
    i for i, bow in enumerate(corpus)
    if len(bow) > 0
]

corpus = [corpus[i] for i in valid_docs]
tokenized_data = [tokenized_data[i] for i in valid_docs]
cleaned_text = [cleaned_text[i] for i in valid_docs]
df = df.iloc[valid_docs].reset_index(drop=True)

if len(dictionary) == 0 or not corpus:
    raise ValueError(
        "The LDA dictionary or corpus is empty. "
        "Try reducing the dictionary filtering thresholds."
    )

print("LDA vocabulary size:", len(dictionary))
print("Documents used for LDA:", len(corpus))

num_lda_topics = 5

lda_model = gensim.models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_lda_topics,
    random_state=42,
    passes=10,
    iterations=100
)

print("\n========== LDA TOPICS ==========")

for topic_id, topic in lda_model.print_topics(num_words=10):
    print(f"Topic {topic_id}: {topic}")

# Prepare LDA visualization using metric MDS
lda_display = pyLDAvis.gensim_models.prepare(
    lda_model,
    corpus,
    dictionary,
    sort_topics=False,
    mds='mmds'
)
# Define the output directory
output_dir = BASE_DIR

# Save interactive visualization
html_path = output_dir / 'lda_visualization.html'

pyLDAvis.save_html(lda_display, str(html_path))

print(f"LDA visualization saved to: {html_path}")

# Open in the default browser
webbrowser.open(html_path.resolve().as_uri())


# Save visualization as HTML instead of calling display()
output_dir = BASE_DIR
output_dir.mkdir(parents=True, exist_ok=True)

html_path = output_dir / 'lda_visualization.html'

pyLDAvis.save_html(lda_display, str(html_path))

print(f"\nLDA visualization saved to: {html_path}")

# Open the visualization in your default browser
webbrowser.open(html_path.resolve().as_uri())

num_nmf_topics = 8

nmf_model = NMF(
    n_components=num_nmf_topics,
    random_state=42,
    max_iter=400
)

W = nmf_model.fit_transform(tfidf_matrix)
H = nmf_model.components_

feature_names = tfidf_vectorizer.get_feature_names_out()

print("\n========== NMF TOPICS ==========")

for topic_idx, topic in enumerate(H):
    top_indices = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_indices]

    print(f"\nTopic {topic_idx}:")
    print(", ".join(top_words))

for topic_idx, topic in enumerate(H):
    top_indices = topic.argsort()[:-21:-1]
    top_words = [feature_names[i] for i in top_indices]

    print(f"\nWordCloud for Topic {topic_idx}")

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color='white',
        collocations=False
    ).generate(" ".join(top_words))

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'NMF Topic {topic_idx}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
