# ArXplorer
Recommender of daily papers from arXiv, customized with your Prompt. Minimal, hackable, no-boilerplate.

<div align="center">
  <img src="https://github.com/billxbf/arxplorer/assets/65674752/5b4f6728-da3e-4891-9d88-ef33f1176fea" width=400 height=400>
  <br>
  <p>"I like innovative papers in large foundation models, multimodal methods, symbolic reasoning and automation."</p>
</div>

## What's up
Now we've been overwhelmed by papers on arXiv. With ~300 new additions **daily** in [cs.AI](https://arxiv.org/list/cs.AI/new) section alone, sifting through them can be daunting. 
This project scrapes daily feed from https://arxiv.org/list/{namespace}/new, collecting author data and performing two-stage ranking:
- *Coarse Ranking*: Use the authors' impact index and a CPU-friendly embedding [model](https://huggingface.co/WhereIsAI/UAE-Large-V1) (per [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) 🤗) to reduce candidate pools into ~20 by weighted Copeland ranking. 
- *Reranking*: Optionally use gpt-4 to choose top k and write a summary (which is cheap for just one call per day).

## Quick Start
Prepare environment
```
conda create -n "arxplorer" python==3.11
conda activate arxplorer
pip install -r requirements.txt
```
(Recommended) Use an OpenAI key for summarization and better ranking.
```
echo 'OPENAI_API_KEY=your_api_key_here' >> .env
```
GO!
```
python run.py
```

## Customization
You may customize your preferences or interests by 
```
echo 'INSTRUCTION="I like ..."' >> .env
```
Use `namespace` to specify the section in arXiv to scrape from (make sure https://arxiv.org/list/{namespace}/new can be visited). Use `top_k` to specify the final number of feeds you want to see. `coarse_k` is the intermediate number from coarse ranking and should always be larger than `top_k`.
```
python run.py --namespace="cs.AI" --top_k=10 --coarse_k=20
```
`fast_mode` is set to True by default, which ignores author-related features. Collecting author data stably (using [scholarly](https://github.com/scholarly-python-package/scholarly) and [free-proxy](https://github.com/jundymek/free-proxy) can be painfully slow to start with (while going better as `authors_cache.db` automatically builds up the cache). If you are deploying on server or have ~1hr to let it run, 
```
python run.py --fast_mode=False
```

## Disclaimer 
This ranker is soooo biased and I'm pretty sure some cool papers are missed. But I feel it helpful in capturing part of the works I regret to miss.

## Next Step 
I'll create a Tweeter Bot soon to serve this project into daily feed. Feel free to contact me @billxbf for suggestions or contribute to more features, faster pipelines etc :) 

