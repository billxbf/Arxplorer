import logging
import sqlite3
from typing import List

import requests
from bs4 import BeautifulSoup
from scholarly import scholarly
from tqdm import tqdm

from arxplorer.datamodel import Author, Feed
from arxplorer.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

MAX_AUTHORS = 2  # Maximum number of authors to investigate, for speedup and regularizing "hotpot" papers.

def parse_arxiv(namespace: str, fast_mode: bool = False, n_tests: int = 999) -> List[Feed]:
    url = f"https://arxiv.org/list/{namespace}/new"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    papers = soup.find_all('dt')
    
    print(f'Found {len(papers)} papers in the {namespace} section of arXiv.\n')

    feeds = []
    n_blocks = min(len(papers), n_tests)
    for i in tqdm(range(n_blocks), desc="Collecting author data ..."):
        block = papers[i]

        try:
            # Find the <dd> tag that immediately follows each <dt> tag
            metadata = block.find_next_sibling('dd')
            title = metadata.find('div', class_='list-title').text.replace('Title:', '').strip()
            author_names = [a.text for a in metadata.find('div', class_='list-authors').find_all('a')][:MAX_AUTHORS]
            authors = [_parse_scholar(author_name, fast_mode) for author_name in author_names]
            abstract = metadata.find('p').text.strip()
            # Extract the PDF link from the <dt> block
            pdf_link_suffix = block.find('a', title='Download PDF')['href']
            pdf_url = f'https://arxiv.org{pdf_link_suffix}'

            feed = Feed(section=namespace, 
                        pdf_url=pdf_url, 
                        title=title, 
                        authors=authors, 
                        f_author=authors[0], 
                        abstract=abstract, 
                        summary=None)
            
            feeds.append(feed)
            # time.sleep(1)

        except Exception as e:
            logger.info(e)
            logger.info('Error extracting information for one of the papers. Skipping to the next.\n')
            continue

    return feeds


# Function to get author from cache or fetch and cache if not exists
def _parse_scholar(author_name: str, fast_mode: bool = False) -> Author:
    
    if fast_mode:
        logger.info(f'Fast mode enabled. Skipping author data retrieval for {author_name}')
        return Author(author_name, '', '', 0, 0, 0)

    logger.info(f'Retrieving Author data for {author_name}')

    # DB init
    conn = sqlite3.connect('authors_cache.db')
    cursor = conn.cursor()

    # Check if the author is already in the cache
    cursor.execute('SELECT * FROM authors WHERE name = ?', (author_name,))
    author_data = cursor.fetchone()

    if author_data:
        logger.info(f'Found cached information for {author_name}')
        
        return Author(*author_data)
    else:
        logger.info(f'Parsing information for {author_name} from Google Scholar...')
        
        try:
            # logger.info(f'Failed to retrieve information for {author_name}. Trying again with a free proxy...')
            # pg = ProxyGenerator()
            # success = pg.FreeProxies(wait_time=20)
            search_query = scholarly.search_author(author_name)
            author = next(search_query)
            scholarly.fill(author, sections=['basics', 'indices', 'counts', 'publications'])
            author_record = (
                author.get('name', ''),
                author.get('affiliation', ''),
                ','.join(author.get('interests', [])),  
                author.get('citedby', 0),
                author.get('hindex', 0),
                len(author.get('publications', [])),
            )
            cursor.execute('INSERT INTO authors VALUES (?, ?, ?, ?, ?, ?)', author_record)
            conn.commit()
            return Author(*author_record)
        except:
            logger.info(f'Failed to retrieve information for {author_name}. Creating a placeholder record...')
            
            author_record = (author_name, '', '', 0, 0, 0)
            # cursor.execute('INSERT INTO authors VALUES (?, ?, ?, ?, ?, ?)', author_record)
            conn.commit()
            return Author(*author_record)
        finally:
            conn.close()