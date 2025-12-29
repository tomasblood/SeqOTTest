"""
Real-world data loading for NeurIPS/ArXiv papers

Supports:
- NeurIPS papers (1987-2019)
- ArXiv CS subset
- Pre-computed embeddings or on-the-fly embedding generation
"""

import numpy as np
import os
import json
import pickle
from collections import defaultdict
from datetime import datetime


class NeurIPSDataLoader:
    """
    Load NeurIPS papers with embeddings.

    Data can come from:
    1. Pre-computed embeddings (preferred)
    2. Raw text data + embedding model
    3. Kaggle NeurIPS dataset

    Parameters
    ----------
    data_dir : str
        Directory containing the data
    year_range : tuple of int
        (start_year, end_year) for papers to include
    min_papers_per_year : int
        Minimum number of papers per year to include that year
    """

    def __init__(self, data_dir='data/neurips', year_range=(2012, 2022),
                 min_papers_per_year=50):
        self.data_dir = data_dir
        self.year_range = year_range
        self.min_papers_per_year = min_papers_per_year

        self.papers = None
        self.embeddings_by_year = None
        self.metadata_by_year = None

    def load_from_pickle(self, pickle_path):
        """
        Load pre-computed embeddings from pickle file.

        Expected format:
        {
            'embeddings_by_year': {year: np.ndarray, ...},
            'metadata_by_year': {year: [metadata_dicts], ...}
        }
        """
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        self.embeddings_by_year = data['embeddings_by_year']
        self.metadata_by_year = data.get('metadata_by_year', {})

        # Filter by year range
        self.embeddings_by_year = {
            year: emb for year, emb in self.embeddings_by_year.items()
            if self.year_range[0] <= year <= self.year_range[1]
        }

        self.metadata_by_year = {
            year: meta for year, meta in self.metadata_by_year.items()
            if self.year_range[0] <= year <= self.year_range[1]
        }

        print(f"Loaded embeddings for years: {sorted(self.embeddings_by_year.keys())}")

        return self

    def load_from_csv(self, csv_path, text_column='abstract',
                     year_column='year', title_column='title'):
        """
        Load papers from CSV and compute embeddings.

        Parameters
        ----------
        csv_path : str
            Path to CSV file
        text_column : str
            Column containing text to embed
        year_column : str
            Column containing year
        title_column : str
            Column containing paper title
        """
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Filter by year range
        df = df[
            (df[year_column] >= self.year_range[0]) &
            (df[year_column] <= self.year_range[1])
        ]

        # Group by year
        papers_by_year = defaultdict(list)
        for _, row in df.iterrows():
            year = int(row[year_column])
            papers_by_year[year].append({
                'text': row[text_column],
                'title': row.get(title_column, ''),
                'year': year
            })

        # Filter years with too few papers
        papers_by_year = {
            year: papers for year, papers in papers_by_year.items()
            if len(papers) >= self.min_papers_per_year
        }

        print(f"Found papers for years: {sorted(papers_by_year.keys())}")
        for year, papers in sorted(papers_by_year.items()):
            print(f"  {year}: {len(papers)} papers")

        self.papers = papers_by_year
        self.metadata_by_year = papers_by_year

        return self

    def compute_embeddings(self, method='tfidf', **kwargs):
        """
        Compute embeddings from text.

        Parameters
        ----------
        method : str
            'tfidf', 'sentence-transformers', or 'openai'
        **kwargs : dict
            Additional arguments for embedding method
        """
        if self.papers is None:
            raise ValueError("Must load papers first using load_from_csv()")

        from sklearn.feature_extraction.text import TfidfVectorizer

        if method == 'tfidf':
            # Collect all texts
            all_texts = []
            year_indices = []

            for year in sorted(self.papers.keys()):
                texts = [p['text'] for p in self.papers[year]]
                all_texts.extend(texts)
                year_indices.append((year, len(texts)))

            # Compute TF-IDF
            print("Computing TF-IDF embeddings...")
            vectorizer = TfidfVectorizer(
                max_features=kwargs.get('max_features', 500),
                min_df=kwargs.get('min_df', 2),
                max_df=kwargs.get('max_df', 0.8),
                stop_words='english'
            )

            embeddings = vectorizer.fit_transform(all_texts).toarray()

            # Split by year
            self.embeddings_by_year = {}
            start_idx = 0
            for year, n_papers in year_indices:
                self.embeddings_by_year[year] = embeddings[start_idx:start_idx + n_papers]
                start_idx += n_papers

            print(f"Computed embeddings with {embeddings.shape[1]} dimensions")

        elif method == 'sentence-transformers':
            from sentence_transformers import SentenceTransformer

            model_name = kwargs.get('model_name', 'all-MiniLM-L6-v2')
            print(f"Loading Sentence-BERT model: {model_name}")
            model = SentenceTransformer(model_name)

            self.embeddings_by_year = {}
            for year in sorted(self.papers.keys()):
                texts = [p['text'] for p in self.papers[year]]
                print(f"  Encoding {len(texts)} papers from {year}...")
                embeddings = model.encode(texts, show_progress_bar=True)
                self.embeddings_by_year[year] = embeddings

        else:
            raise ValueError(f"Unknown method: {method}")

        return self

    def get_sequential_embeddings(self):
        """
        Get embeddings as a list ordered by year.

        Returns
        -------
        embeddings : list of np.ndarray
            Embeddings for each year
        years : list of int
            Corresponding years
        metadata : list of list of dict
            Metadata for each paper
        """
        years = sorted(self.embeddings_by_year.keys())
        embeddings = [self.embeddings_by_year[year] for year in years]
        metadata = [self.metadata_by_year.get(year, []) for year in years]

        return embeddings, years, metadata

    def save_embeddings(self, output_path):
        """Save embeddings to pickle file."""
        data = {
            'embeddings_by_year': self.embeddings_by_year,
            'metadata_by_year': self.metadata_by_year,
            'year_range': self.year_range
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved embeddings to {output_path}")


class ArXivDataLoader:
    """
    Load ArXiv CS papers with embeddings.

    Similar interface to NeurIPSDataLoader but for ArXiv data.
    """

    def __init__(self, data_dir='data/arxiv', year_range=(2015, 2023),
                 min_papers_per_year=100, categories=None):
        self.data_dir = data_dir
        self.year_range = year_range
        self.min_papers_per_year = min_papers_per_year
        self.categories = categories or ['cs.LG', 'cs.AI', 'cs.CL', 'cs.CV']

        self.embeddings_by_year = None
        self.metadata_by_year = None

    def load_from_pickle(self, pickle_path):
        """Load pre-computed embeddings from pickle file."""
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        self.embeddings_by_year = data['embeddings_by_year']
        self.metadata_by_year = data.get('metadata_by_year', {})

        # Filter by year range
        self.embeddings_by_year = {
            year: emb for year, emb in self.embeddings_by_year.items()
            if self.year_range[0] <= year <= self.year_range[1]
        }

        return self

    def get_sequential_embeddings(self):
        """Get embeddings as a list ordered by year."""
        years = sorted(self.embeddings_by_year.keys())
        embeddings = [self.embeddings_by_year[year] for year in years]
        metadata = [self.metadata_by_year.get(year, []) for year in years]

        return embeddings, years, metadata


def create_sample_neurips_data(output_path='data/neurips/sample_embeddings.pkl',
                               n_years=6, n_papers_per_year=200, n_dims=300,
                               random_state=42):
    """
    Create sample NeurIPS-like data for testing.

    This simulates the evolution of research topics over time.
    """
    rng = np.random.RandomState(random_state)

    start_year = 2015
    embeddings_by_year = {}
    metadata_by_year = {}

    # Define evolving research topics
    topics = {
        'deep_learning': rng.randn(n_dims),
        'transformers': rng.randn(n_dims),
        'gans': rng.randn(n_dims),
        'rl': rng.randn(n_dims),
        'graph_neural_nets': rng.randn(n_dims),
    }

    # Topic prevalence over time
    topic_trends = {
        'deep_learning': [0.4, 0.3, 0.2, 0.2, 0.15, 0.1],
        'transformers': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'gans': [0.2, 0.3, 0.3, 0.2, 0.15, 0.1],
        'rl': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        'graph_neural_nets': [0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
    }

    for year_idx in range(n_years):
        year = start_year + year_idx
        embeddings = []
        metadata = []

        for paper_idx in range(n_papers_per_year):
            # Sample topic based on trends
            topic_probs = [topic_trends[t][year_idx] for t in topics.keys()]
            topic_probs = np.array(topic_probs) / sum(topic_probs)
            topic = rng.choice(list(topics.keys()), p=topic_probs)

            # Create embedding as noisy version of topic center
            embedding = topics[topic] + rng.randn(n_dims) * 0.3

            embeddings.append(embedding)
            metadata.append({
                'title': f'Paper {paper_idx} on {topic}',
                'topic': topic,
                'year': year
            })

        embeddings_by_year[year] = np.array(embeddings)
        metadata_by_year[year] = metadata

    # Save
    data = {
        'embeddings_by_year': embeddings_by_year,
        'metadata_by_year': metadata_by_year,
        'year_range': (start_year, start_year + n_years - 1)
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Created sample NeurIPS data:")
    print(f"  Years: {start_year}-{start_year + n_years - 1}")
    print(f"  Papers per year: {n_papers_per_year}")
    print(f"  Embedding dim: {n_dims}")
    print(f"  Saved to: {output_path}")

    return output_path
