from arxplorer.datamodel import Feed, Config
from typing import List


def first_author_citation(feed: Feed) -> int:
    return feed.authors[0].citation

def avg_authors_citation(feed: Feed) -> float:
    return sum([a.citation for a in feed.authors]) / len(feed.authors)

def variance_authors_citation(feed: Feed) -> float:
    avg = avg_authors_citation(feed)
    return sum([(a.citation - avg) ** 2 for a in feed.authors]) / len(feed.authors)

def first_author_h_index(feed: Feed) -> int:
    return feed.authors[0].h_index

def avg_authors_h_index(feed: Feed) -> float:
    return sum([a.h_index for a in feed.authors]) / len(feed.authors)

def variance_authors_h_index(feed: Feed) -> float:
    avg = avg_authors_h_index(feed)
    return sum([(a.h_index - avg) ** 2 for a in feed.authors]) / len(feed.authors)

def embedding_L2_similarity(e1: List[float], e2: List[float]) -> float:
    return 1 / sum([(x - y) ** 2 for x, y in zip(e1, e2)]) ** 0.5

def institution_ranking():
    pass

def author_white_list():
    pass

def topic_white_list():
    pass

#TODO: Add more later when data comes in 