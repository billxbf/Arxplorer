import argparse
from arxplorer.parsers import parse_arxiv
from arxplorer.ranker import PaperRanker
from arxplorer.datamodel import Config
from arxplorer.utils import print_paper_metadata

# Set up argument parser
parser = argparse.ArgumentParser(description='Rank daily paper feeds from arXiv.')
parser.add_argument('--namespace', type=str, default='cs.AI', help='arXiv namespace to parse and rank')
parser.add_argument('--instruction', type=str, default='I like papers with innovations in model architectures, multimodal methods, symbolic reasoning and automation. Others general ML topics are welcome, while direct sub-field applications are less interesting.', help='Instruction for OpenAI to rerank and summarize')
parser.add_argument('--top_k', type=int, default=10, help='Number of top papers to select')
parser.add_argument('--coarse_k', type=int, default=20, help='Number of papers for coarse ranking')
parser.add_argument('--fast_mode', type=bool, default=True, help='Whether to scrape Author info')
parser.add_argument('--use_openai', type=bool, default=True, help='Whether to use OpenAI to rerank and summarize')

args = parser.parse_args()

# Use arguments to parse feeds and configure ranking
feeds = parse_arxiv(args.namespace, fast_mode=args.fast_mode)

cfg = Config(top_k=args.top_k, 
             coarse_k=args.coarse_k, 
             instruction=args.instruction, 
             namespace=args.namespace, 
             use_openai=args.use_openai)

ranker = PaperRanker(cfg)
ranked_feeds = ranker.rank(feeds)

print_paper_metadata(ranked_feeds)