from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Author:
    name: str
    affiliation: str 
    interests: str 
    citation: int 
    h_index: int
    n_publications: int

@dataclass
class Feed:
    section: str 
    pdf_url: str
    title: str
    authors: List[Author]
    f_author: Author
    abstract: str
    summary: Optional[str] 


@dataclass
class Config:
    namespace: str = "cs.AI"
    instruction: str = "I like papers with innovative ideas and promising applications."
    top_k: int = 10  
    coarse_k: int = 20  # Number of papers to keep after coarse ranking
    use_openai: bool = True  # Whether to use OpenAI to rerank and summarize
    
    
