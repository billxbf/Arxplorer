import json
import logging
from typing import Dict, List

import torch
from angle_emb import AnglE, Prompts

from arxplorer.datamodel import Config, Feed
from arxplorer.logging import setup_logging
from arxplorer.utils import *

setup_logging()
logger = logging.getLogger(__name__)


# weight assigned for copeland ranking
# higher weight means higher importance
FEATURE_WEIGHT = {"f_author_citation_rank": 0.4, 
                  "f_author_h_index_rank": 0.4,
                  "avg_author_citation_rank": 0.4,  # we are scraping the first author only at this time
                  "avg_author_h_index_rank": 0.4,
                  "f_author_interests_match_rank": 1.0,
                  "abstract_match_rank": 1.4}

device = "cuda" if torch.cuda.is_available() else "cpu"

class PaperRanker:
    def __init__(self, config: Config):

        angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to(device)
        angle.set_prompt(prompt=Prompts.C)

        self.config = config
        # Best CPU friendly embedding model per https://huggingface.co/spaces/mteb/leaderboard
        self.tiny_lm = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to(device)
        self.tiny_lm.set_prompt(prompt=Prompts.C)
        self.openai_model = "gpt-4"
        self.instruction_embedding = self.tiny_lm.encode({'text': self.config.instruction}, 
                                                         to_numpy=True, device=device)

    def rank(self, feeds: List[Feed], return_json: bool = False):
        # The plan here is to use simple and "cheap" signals for coarse first-round ranking,
        # reducing ~300 papers into ~30. Then a more expensive LLM to rerank and generate summaries.
        coarse_ranked = self._coarse_ranking(feeds)
        
        reranked = self._rerank(coarse_ranked)

        if return_json:
            reranked = [feed.to_json() for feed in reranked]
        
        return reranked
    
    def _encode(self, text: str) -> List[float]:
        return self.tiny_lm.encode({'text': text}, to_numpy=True, device=device)


    def _coarse_ranking(self, feeds: List[Feed]) -> List[Feed]:
        
        # "cheap" features
        f_author_citation_rank = sorted(feeds, key=lambda x: first_author_citation(x), reverse=True)
        f_author_h_index_rank = sorted(feeds, key=lambda x: first_author_h_index(x), reverse=True)
        avg_author_citation_rank = sorted(feeds, key=lambda x: avg_authors_citation(x), reverse=True)
        avg_author_h_index_rank = sorted(feeds, key=lambda x: avg_authors_h_index(x), reverse=True)
        f_author_interests_match_rank = sorted(feeds, 
                                         key=lambda x: embedding_L2_similarity(self._encode(x.f_author.interests), 
                                                                               self.instruction_embedding),
                                         reverse=True)
        abstract_match_rank = sorted(feeds, 
                                     key=lambda x: embedding_L2_similarity(self._encode(x.abstract), 
                                                                           self.instruction_embedding),
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
            feeds,
        )
        
        logger.info(f'Coarse Rank result: {ranked}')

        return ranked[:self.config.coarse_k]
    
    def _rerank(self, feeds: List[Feed]) -> List[Feed]:

        if not self.config.use_openai:
            return feeds[:self.config.top_k]
            
        # using openai API since I'm expecting small traffic (1 call per day)
        feeds_prompt = json.dumps(
            dict(papers=[{"title": feed.title, "abstract": feed.abstract} for feed in feeds]),
            indent=4
        )
        rerank_prompt = RERANK_PROMPT.format(
            coarse_k=self.config.coarse_k,
            top_k=self.config.top_k,
            namespace=self.config.namespace,
            instruction=self.config.instruction,
            feeds=feeds_prompt
        )

        logger.info(f'Constructed prompt for reranking: {rerank_prompt}')

        try:
            completion = openai_chat_completion(rerank_prompt, self.openai_model)

            logger.info(f'OpenAI completion: {completion}')

            response = json.loads(completion).get('recommendations', [])
            
            reranked = []
            for resp in response:

                feed = next((x for x in feeds if x.title.strip().lower()
                              == resp['title'].strip().lower()), None)
                feed.summary = resp['summary']
                reranked.append(feed)

        except:
            # fallback to coarse ranking
            logger.info('OpenAI API failed. Fallback to coarse ranking.')
            
            reranked = feeds[:self.config.top_k]
        
        return reranked
                
            
    def _weighted_copeland_scores(self, rankings: Dict[str, List[Feed]], weights: Dict[str, float], 
                                  papers: List[Feed]) -> List[Feed]:
        scores = {paper.title: 0 for paper in papers}  
        
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
                        scores[paper1.title] += weight
                        scores[paper2.title] -= weight
                    elif rank1 > rank2:  
                        scores[paper1.title] -= weight
                        scores[paper2.title] += weight

        sorted_papers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        paper_names = [paper_name for paper_name, _ in sorted_papers]

        return [next((x for x in papers if x.title == paper_name), None) for paper_name in paper_names]