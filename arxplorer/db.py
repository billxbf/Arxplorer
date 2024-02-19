import sqlite3
import logging
from arxplorer.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Initialize an Author database to prevent repetitive queries (run this once)
def init_db():

    import os
    if not os.path.exists('authors_cache.db'):
        logger.info('Initializing database...')

        conn = sqlite3.connect('authors_cache.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS authors (
                name TEXT PRIMARY KEY,
                affiliation TEXT,
                interests TEXT,
                citation INTEGER,
                h_index INTEGER,
                n_publications INTEGER
            )
        ''')
        conn.commit()
        conn.close()

        logger.info('Database initialized.')

# Reinitialize Author database.
# Warning: This will completely erase the old one!
def reset_db():

    logger.info('Resetting database...')

    conn = sqlite3.connect('authors_cache.db')
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS authors')
    conn.commit()
    conn.close()

    logger.info('Database dropped. Reinitializing...')

    init_db()

    print('Database reset.')
    logger.info('Database reset.')