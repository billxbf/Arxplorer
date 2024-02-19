import requests
from bs4 import BeautifulSoup
from typing import List
import sqlite3
from arxplorer.datamodel import Feed, Author
from scholarly import scholarly
from tqdm import tqdm

import logging
from arxplorer.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def parse_arxiv(namespace: str, tests: int = 0) -> List[Feed]:
    url = f"https://arxiv.org/list/{namespace}/new"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    papers = soup.find_all('dt')
    
    print(f'Found {len(papers)} papers in the {namespace} section of arXiv.\n')

    count = 0  # for test mode only
    feeds = []
    for block in tqdm(papers, desc="Collecting data from papers ..."):
        try:
            # Find the <dd> tag that immediately follows each <dt> tag
            metadata = block.find_next_sibling('dd')
            title = metadata.find('div', class_='list-title').text.replace('Title:', '').strip()
            author_names = [a.text for a in metadata.find('div', class_='list-authors').find_all('a')]
            authors = [_parse_scholar(author_name) for author_name in author_names]
            abstract = metadata.find('p').text.strip()
            # Extract the PDF link from the <dt> block
            pdf_link_suffix = block.find('a', title='Download PDF')['href']
            pdf_url = f'https://arxiv.org{pdf_link_suffix}'

            feed = Feed(section=namespace, pdf_url=pdf_url, title=title, authors=authors, f_author=authors[0], abstract=abstract)
            feeds.append(feed)

            # you don't want to rerun the whole stuff for testing
            if tests:
                count += 1
                if count >= tests:
                    break
        except:
            logger.info('Error extracting information for one of the papers. Skipping to the next.\n')
            continue

    return feeds


# Function to get author from cache or fetch and cache if not exists
def _parse_scholar(author_name: str) -> Author:

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
            cursor.execute('INSERT INTO authors VALUES (?, ?, ?, ?, ?, ?)', author_record)
            conn.commit()
            return Author(*author_record)
        finally:
            conn.close()