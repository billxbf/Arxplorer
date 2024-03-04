from typing import List
import numpy as np

from arxplorer.datamodel import Feed


def openai_chat_completion(prompt: str, model="gpt-3.5-turbo") -> str:
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that recommend daily papers from arXiv."},
        {"role": "user", "content": prompt},
    ]
    )
    return response.choices[0].message.content

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

def embedding_L2_similarity(vec1: List[float], vec2: List[float]) -> float:
    return -np.linalg.norm(np.array(vec1) - np.array(vec2))

def institution_ranking():
    pass

def author_white_list():
    pass

def topic_white_list():
    pass

#TODO: Add more later on


RERANK_PROMPT = '''
## Task Description
Following are {coarse_k} new papers with abstracts selected from arXiv {namespace} section.
Consider I'm familiar with this research field. Your task is to rank and recommend top {top_k} papers that match my preferences. 
For each of your recommendation, provide a 1~2 sentence summary of the abstract.

## My Preferences
{instruction}

## Papers
{feeds}

## Template for Your Response (must be directly callable by json.loads() method in Python)
{{
    "recommendations": [
        {{
            "title": "Title of the most recommended paper",
            "summary": "1~2 sentence summary of the abstract",
        }},
        {{
            "title": "Title of the second recommended paper",
            "summary": "1~2 sentence summary of the abstract",
        }}
    ]
}}
## Your Response
'''


def wrap_text(text, width):
    words = text.split(' ')
    wrapped_lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            current_line += word + " "
        else:
            wrapped_lines.append(current_line.rstrip())
            current_line = word + " "
    wrapped_lines.append(current_line.rstrip())  # Add last line
    return wrapped_lines

def print_paper_metadata(papers):
    # ANSI color codes
    color_title = "\033[94m"  # Blue
    color_url = "\033[92m"    # Green
    color_summary = "\033[93m" # Yellow
    reset_color = "\033[0m"   # Reset to default terminal color
    
    max_width = 75  # Adjusted for consistent right border
    border_line = "+" + "-" * 78 + "+"
    for paper in papers:
        print(border_line)
        title_lines = wrap_text(paper.title, max_width)
        for line in title_lines:
            print(f"| {color_title}{line.ljust(max_width)}{reset_color}  |")
        url_line = f"| {color_url}URL: {paper.pdf_url}{reset_color}"
        print(f"{url_line.ljust(max_width + 9)}    |")  # Adjust padding for color codes
        print(border_line)
        if paper.summary:
            print(f"| {color_summary}Summary: {reset_color}".ljust(max_width + 9) + "    |")  # Adjust padding for color codes
            summary_lines = wrap_text(paper.summary, max_width)
            for line in summary_lines:
                print(f"| {line.ljust(max_width)}  |")
        print(border_line)
        print()  # Print an empty line for spacing between papers