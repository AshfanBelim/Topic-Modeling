"""Topic modeling of Kaggle dataset descriptions.

Runs two unsupervised topic models on the `Description` column:
  * LDA  (Gensim)              -> interactive pyLDAvis HTML
  * NMF  (TF-IDF + scikit-learn) -> one word cloud per topic

Usage:
    python topic_modeling.py
    python topic_modeling.py --lda-topics 6 --nmf-topics 6
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files; no display window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
SEED = 42
CUSTOM_STOP_WORDS = {"using", "use", "used", "also", "new", "like"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | CUSTOM_STOP_WORDS


#data
def load_data(path, column):
    """Read the CSV and return a clean Series of non-empty descriptions."""
    df = pd.read_csv(path, encoding_errors="replace")
    if column not in df.columns:
        raise SystemExit(
            f"Column '{column}' not found in {path}.\nAvailable columns: {list(df.columns)}"
        )
    texts = df[column].dropna().astype(str).str.strip()
    texts = texts[texts != ""].reset_index(drop=True)
    print(f"Loaded {len(texts)} non-empty descriptions (from {len(df)} rows).")
    return texts


def preprocess(texts):
    """Lowercase, tokenize, strip accents, drop short tokens and stop words."""
    from gensim.utils import simple_preprocess

    docs = []
    for text in texts:
        tokens = [
            w for w in simple_preprocess(text, deacc=True, min_len=2) if w not in STOP_WORDS
        ]
        if tokens:  # drop documents that became empty
            docs.append(tokens)
    print(f"{len(docs)} documents remain after preprocessing.")
    return docs


#LDA
def run_lda(docs, num_topics, top_n, out_dir):
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel, LdaModel
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis

    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above=0.9)

    pairs = [(dictionary.doc2bow(d), d) for d in docs]
    pairs = [(bow, d) for bow, d in pairs if bow]  # drop docs emptied by filtering
    corpus = [bow for bow, _ in pairs]
    texts = [d for _, d in pairs]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        iterations=100,
        random_state=SEED,
    )
    topics = [[w for w, _ in lda.show_topic(i, topn=top_n)] for i in range(num_topics)]

    coherence = None
    try:
        coherence = CoherenceModel(
            model=lda, texts=texts, dictionary=dictionary, coherence="c_v", processes=1
        ).get_coherence()
    except Exception as exc:  # coherence is a bonus metric, never block the run
        print(f"Could not compute coherence: {exc}")

    vis = gensimvis.prepare(lda, corpus, dictionary, mds="mmds")
    html_path = out_dir / "lda_visualization.html"
    pyLDAvis.save_html(vis, str(html_path))
    print(f"Saved interactive LDA visualization: {html_path}")
    return topics, coherence


# NMF
def run_nmf(docs, num_topics, top_n, cloud_n, images_dir):
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf = vectorizer.fit_transform([" ".join(d) for d in docs])
    terms = vectorizer.get_feature_names_out()

    nmf = NMF(n_components=num_topics, max_iter=400, random_state=SEED)
    nmf.fit(tfidf)

    topics, cloud_weights = [], []
    for component in nmf.components_:
        order = component.argsort()[::-1]
        topics.append([terms[i] for i in order[:top_n]])
        cloud_weights.append(
            {terms[i]: float(component[i]) for i in order[:cloud_n] if component[i] > 0}
        )

    save_word_clouds(cloud_weights, images_dir)
    return topics


def save_word_clouds(cloud_weights, images_dir):
    from wordcloud import WordCloud

    clouds = [
        WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(w)
        for w in cloud_weights
    ]

    for i, cloud in enumerate(clouds, start=1):  # one image per topic
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(f"NMF Topic {i}")
        ax.axis("off")
        fig.savefig(images_dir / f"nmf_topic_{i}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    cols = 2  # one overview image with every topic
    rows = math.ceil(len(clouds) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in axes:
        ax.axis("off")
    for i, (ax, cloud) in enumerate(zip(axes, clouds), start=1):
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(f"Topic {i}")
    fig.tight_layout()
    fig.savefig(images_dir / "nmf_topics_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {len(clouds)} word clouds to {images_dir}")


# Reporting
def write_summary(path, n_docs, lda_topics, coherence, nmf_topics):
    lines = ["# Topic modeling results", "", f"Documents modeled: **{n_docs}**", ""]
    lines += ["## LDA topics", ""]
    if coherence is not None:
        lines += [f"Coherence (c_v): **{coherence:.3f}**", ""]
    for i, words in enumerate(lda_topics, start=1):
        lines.append(f"- **Topic {i}:** {', '.join(words)}")
    lines += ["", "## NMF topics", ""]
    for i, words in enumerate(nmf_topics, start=1):
        lines.append(f"- **Topic {i}:** {', '.join(words)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved summary: {path}")


def print_topics(title, topics):
    print(f"\n=== {title} ===")
    for i, words in enumerate(topics, start=1):
        print(f"Topic {i}: {', '.join(words)}")


# Main
def parse_args():
    p = argparse.ArgumentParser(description="LDA and NMF topic modeling on Kaggle descriptions.")
    p.add_argument("--data", default=str(BASE_DIR / "data" / "voted-kaggle-dataset.csv"))
    p.add_argument("--text-column", default="Description")
    p.add_argument("--lda-topics", type=int, default=5)
    p.add_argument("--nmf-topics", type=int, default=8)
    p.add_argument("--top-words", type=int, default=10)
    p.add_argument("--cloud-words", type=int, default=20)
    p.add_argument("--out-dir", default=str(BASE_DIR / "outputs"))
    p.add_argument("--images-dir", default=str(BASE_DIR / "images"))
    return p.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():  # fall back to a CSV placed next to the script
        fallback = BASE_DIR / "voted-kaggle-dataset.csv"
        if fallback.exists():
            data_path = fallback
        else:
            raise SystemExit(f"Dataset not found: {data_path}. Put the CSV in the data/ folder.")

    out_dir, images_dir = Path(args.out_dir), Path(args.images_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    texts = load_data(data_path, args.text_column)
    docs = preprocess(texts)

    lda_topics, coherence = run_lda(docs, args.lda_topics, args.top_words, out_dir)
    print_topics("LDA topics", lda_topics)
    if coherence is not None:
        print(f"LDA coherence (c_v): {coherence:.3f}")

    nmf_topics = run_nmf(docs, args.nmf_topics, args.top_words, args.cloud_words, images_dir)
    print_topics("NMF topics", nmf_topics)

    write_summary(out_dir / "topics_summary.md", len(docs), lda_topics, coherence, nmf_topics)


if __name__ == "__main__":
    main()
