from arxplorer.datamodel import Feed, Config, Author
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from arxplorer.utils import *

# weight assigned for copeland ranking
# higher weight means higher importance
FEATURE_WEIGHT = {"f_author_citation_rank": 0.8, 
                  "f_author_h_index_rank": 1.0,
                  "avg_author_citation_rank": 0.6,
                  "avg_author_h_index_rank": 0.8,
                  "f_author_interests_match_rank": 1.0,
                  "abstract_match_rank": 1.4}


class PaperRanker:
    def __init__(self, config: Config):
        self.config = config
        self.tiny_lm = SentenceTransformer("all-MiniLM-L6-v2")
        self.instruction_embedding = self.tiny_lm.encode(self.config.instruction)

    def rank(self, feeds: List[Feed]) -> List[Feed]:
        # the plan here is to use simple and "cheap" signals for coarse ranking,
        # reducing ~300 papers into ~30. Then use a more expensive LLM to rerank for top 10.
        coarse_ranked = self._coarse_ranking(feeds)
        


    def _coarse_ranking(self, feeds: List[Feed]) -> List[Feed]:
        
        # "cheap" features
        f_author_citation_rank = sorted(feeds, key=lambda x: first_author_citation(x), reverse=True)
        f_author_h_index_rank = sorted(feeds, key=lambda x: first_author_h_index(x), reverse=True)
        avg_author_citation_rank = sorted(feeds, key=lambda x: avg_authors_citation(x), reverse=True)
        avg_author_h_index_rank = sorted(feeds, key=lambda x: avg_authors_h_index(x), reverse=True)
        f_author_interests_match_rank = sorted(feeds, 
                                         key=lambda x: embedding_L2_similarity(x.f_author.interests, self.instruction_embedding),
                                         reverse=True)
        abstract_match_rank = sorted(feeds, 
                                     key=lambda x: embedding_L2_similarity(x.abstract, self.instruction_embedding),
                                     reverse=True)
        
        # since we don't have any target data for now, let's use weighted Copeland score to combine these rankings
        ranked = self._weighted_copeland_scores(
            {"f_author_citation_rank": f_author_citation_rank,
             "f_author_h_index_rank": f_author_h_index_rank,
             "avg_author_citation_rank": avg_author_citation_rank,
             "avg_author_h_index_rank": avg_author_h_index_rank,
             "f_author_interests_match_rank": f_author_interests_match_rank,
             "abstract_match_rank": abstract_match_rank},
            FEATURE_WEIGHT,
            feeds
        )

        return ranked[:self.config.coarse_k]
        

    def _weighted_copeland_scores(rankings: Dict[str, List[Feed]], weights: Dict[str, float], 
                                  papers: List[Feed]) -> List[Feed]:
        scores = {paper: 0 for paper in papers}  # Initialize scores for each paper
        
        def get_rank(paper, feature_rankings):
            return feature_rankings.index(paper)
        
        # Perform pairwise comparisons
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                paper1, paper2 = papers[i], papers[j]
                for feature, ranking in rankings.items():
                    rank1, rank2 = get_rank(paper1, ranking), get_rank(paper2, ranking)
                    weight = weights[feature]
                    if rank1 < rank2:  
                        scores[paper1] += weight
                        scores[paper2] -= weight
                    elif rank1 > rank2:  
                        scores[paper1] -= weight
                        scores[paper2] += weight

        sorted_papers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [paper for paper, _ in sorted_papers]


    def