import pandas as pd
import numpy as np
import networkx

def read_wikispeedia_graph() -> networkx.Graph:
    wikispeedia = networkx.read_edgelist('datasets/wikispeedia_paths-and-graph/links.tsv',
                                         create_using=networkx.DiGraph)
    return wikispeedia

def read_finished_paths() -> pd.DataFrame:
    paths_finished = pd.read_csv('datasets/wikispeedia_paths-and-graph/paths_finished.tsv', sep='\t', skiprows=15,
                                 names=['hashedIpAddress', 'timestamp', "durationInSec", 'path', "rating"])
    paths_finished['first_article'] = paths_finished['path'].apply(lambda x: x.split(';')[0])
    paths_finished['last_article'] = paths_finished['path'].apply(lambda x: x.split(';')[-1])
    paths_finished['path_length'] = paths_finished['path'].apply(lambda x: len(x.split(';')))
    paths_finished['date'] = pd.to_datetime(paths_finished['timestamp'], unit='s')
    return paths_finished

def read_unfinished_paths() -> pd.DataFrame:
    paths_unfinished = pd.read_csv('datasets/wikispeedia_paths-and-graph/paths_unfinished.tsv', sep='\t', skiprows=16,
                                   names=['hashedIpAddress', 'timestamp', "durationInSec", 'path', "target", "type"])

    return paths_unfinished

def read_articles() -> pd.DataFrame:
    articles = pd.read_csv('datasets/wikispeedia_paths-and-graph/articles.tsv', sep='\t', skiprows=12, header=None, names=['articles'])
    return articles

def read_categories() -> pd.DataFrame:
    categories = pd.read_csv('datasets/wikispeedia_paths-and-graph/categories.tsv', sep='\t', skiprows=12, names = ['article', 'categories'])
    return categories


def read_shortest_path_df() -> pd.DataFrame:
    """Reads in the shortest path matrix. In this method, if there is no path between two
    nodes then the matrix returns a -1"""
    shortest_path = np.genfromtxt("datasets/wikispeedia_paths-and-graph/shortest-path-distance-matrix.txt",
                                  delimiter=1, missing_values=-1, dtype=int)
    articles = pd.read_csv('datasets/wikispeedia_paths-and-graph/articles.tsv', sep='\t', skiprows=12,
                           names=["article_name"])

    shortest_path_df = pd.DataFrame(shortest_path, index=articles.values, columns=articles.values)
    return shortest_path_df

def plaintext_article_finder(article_name: str) -> str:
    art_file_name = "datasets/plaintext_articles/" + article_name + ".txt"
    text_file = open(art_file_name, "r", encoding="utf8")
    res_string = text_file.read()
    text_file.close()

    return res_string

def source_target_paths_information() -> pd.DataFrame:
    """This was entirely written by Sophea, I just took some time to clean it up and make it it's own
    separate method."""
    finished_paths = read_finished_paths()

    article_combinations_count = finished_paths.groupby(['first_article', 'last_article']).size().reset_index(name='count')

    # The mean and std of the path length for each pair of articles
    article_combinations_stats = finished_paths.groupby(['first_article', 'last_article'])['path_length'].agg(['mean', 'std']).reset_index()
    article_combinations_stats['std'] = article_combinations_stats['std'].fillna(0)
    article_combinations_stats.rename(columns={'mean': 'mean_length', 'std': 'std_length'}, inplace=True)

    # The mean and std of the rating for each pair of articles.
    # Note that mean and std may be nan if there are nan ratings. We purposely leave them as nan, as we don't want to fill them with 0s or 1s.
    # Depending on the application, we could change this in the future if neeeded.
    rating_combinations_stats_rating = finished_paths.groupby(['first_article', 'last_article'])['rating'].agg(['mean', 'std']).reset_index()
    mask = rating_combinations_stats_rating['mean'].notnull()
    rating_combinations_stats_rating.loc[mask, 'std'] = rating_combinations_stats_rating.loc[mask, 'std'].fillna(0)
    rating_combinations_stats_rating.rename(columns={'mean': 'mean_rating', 'std': 'std_rating'}, inplace=True)

    # The mean and std of the time for each pair of articles.
    rating_combinations_stats_time = finished_paths.groupby(['first_article', 'last_article'])['durationInSec'].agg(['mean', 'std']).reset_index()
    rating_combinations_stats_time['std'] = rating_combinations_stats_time['std'].fillna(0)
    rating_combinations_stats_time.rename(columns={'mean': 'mean_durationInSec', 'std': 'std_durationInSec'}, inplace=True)

    # Merging all the dataframes
    article_combinations = pd.merge(article_combinations_count, article_combinations_stats, on=['first_article', 'last_article'])
    article_combinations = pd.merge(article_combinations, rating_combinations_stats_rating, on=['first_article', 'last_article'])
    article_combinations = pd.merge(article_combinations, rating_combinations_stats_time, on=['first_article', 'last_article'])

    return article_combinations
