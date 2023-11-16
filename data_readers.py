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