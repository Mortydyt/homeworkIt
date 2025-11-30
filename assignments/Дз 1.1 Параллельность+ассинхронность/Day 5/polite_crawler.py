"""
Days 4-7 - Complete Polite Crawler Implementation
Rate limiting, robots.txt compliance, data storage, and optimization
"""

import asyncio
import aiohttp
import aiofiles
import time
import logging
import re
import json
import csv
import sqlite3
import aiosqlite
import random
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from pathlib import Path
import hashlib

# Import previous modules
import sys
sys.path.append(str(Path(__file__).parent / "Day 3"))
from advanced_crawler import AdvancedCrawler, CrawlStats, CrawlerQueue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('polite_crawler.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class CrawlConfig:
    """Configuration for crawler behavior"""
    max_concurrent: int = 5
    max_per_domain_concurrent: int = 2
    requests_per_second: float = 1.0
    min_delay: float = 1.0
    max_delay: float = 10.0
    jitter: bool = True
    respect_robots: bool = True
    user_agent: str = "PoliteCrawler/1.0 (Educational Purpose)"
    rotate_user_agent: bool = False
    retry_attempts: int = 3
    retry_delay: float = 5.0
    max_depth: int = 3
    same_domain_only: bool = True
    storage_type: str = "json"  # json, csv, sqlite


class RateLimiter:
    """Rate limiting with per-domain and global controls"""

    def __init__(self, requests_per_second: float = 1.0, per_domain: bool = True):
        """
        Initialize rate limiter

        Args:
            requests_per_second: Maximum requests per second
            per_domain: If True, limit per domain instead of globally
        """
        self.requests_per_second = requests_per_second
        self.per_domain = per_domain
        self.min_interval = 1.0 / requests_per_second

        # Rate limiting storage
        if per_domain:
            self.last_request: Dict[str, float] = {}
        else:
            self.last_request: float = 0.0

        self.lock = asyncio.Lock()

        logger.info(f"RateLimiter initialized: {requests_per_second} req/s, per_domain={per_domain}")

    async def acquire(self, domain: Optional[str] = None) -> None:
        """
        Acquire permission to make a request

        Args:
            domain: Domain name for per-domain limiting
        """
        async with self.lock:
            now = time.time()

            if self.per_domain and domain:
                last_time = self.last_request.get(domain, 0.0)
                elapsed = now - last_time

                if elapsed < self.min_interval:
                    wait_time = self.min_interval - elapsed
                    logger.debug(f"Rate limiting {domain}: waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)

                self.last_request[domain] = time.time()

            else:
                elapsed = now - self.last_request
                if elapsed < self.min_interval:
                    wait_time = self.min_interval - elapsed
                    logger.debug(f"Global rate limiting: waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)

                self.last_request = time.time()


class RobotsParser:
    """robots.txt parser with caching"""

    def __init__(self, user_agent: str = "*"):
        """
        Initialize robots parser

        Args:
            user_agent: User agent for robots.txt rules
        """
        self.user_agent = user_agent
        self.cache: Dict[str, RobotFileParser] = {}
        self.cache_lock = asyncio.Lock()
        self.session: Optional[aiohttp.ClientSession] = None

        logger.info(f"RobotsParser initialized for user_agent: {user_agent}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={'User-Agent': self.user_agent}
            )
        return self.session

    async def fetch_robots(self, base_url: str) -> Optional[RobotFileParser]:
        """
        Fetch and parse robots.txt for a domain

        Args:
            base_url: Base URL of the domain

        Returns:
            RobotFileParser instance or None if failed
        """
        parsed = urlparse(base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        async with self.cache_lock:
            if robots_url in self.cache:
                return self.cache[robots_url]

        try:
            session = await self._get_session()
            async with session.get(robots_url) as response:
                if response.status == 200:
                    content = await response.text()

                    # Create and configure robot parser
                    rp = RobotFileParser()
                    rp.set_url(robots_url)
                    rp.parse(content.splitlines())

                    # Cache the result
                    async with self.cache_lock:
                        self.cache[robots_url] = rp

                    logger.info(f"✅ Loaded robots.txt for {parsed.netloc}")
                    return rp
                else:
                    logger.warning(f"⚠️ robots.txt not found for {parsed.netloc} (HTTP {response.status})")
                    return None

        except Exception as e:
            logger.warning(f"❌ Error fetching robots.txt for {parsed.netloc}: {str(e)}")
            return None

    def can_fetch(self, url: str, robots_parser: Optional[RobotFileParser]) -> bool:
        """
        Check if URL can be fetched according to robots.txt

        Args:
            url: URL to check
            robots_parser: RobotFileParser instance

        Returns:
            True if URL can be fetched
        """
        if not robots_parser:
            return True  # No robots.txt rules available

        return robots_parser.can_fetch(self.user_agent, url)

    def get_crawl_delay(self, robots_parser: Optional[RobotFileParser]) -> Optional[float]:
        """
        Get crawl delay from robots.txt

        Args:
            robots_parser: RobotFileParser instance

        Returns:
            Crawl delay in seconds or None
        """
        if not robots_parser:
            return None

        return robots_parser.crawl_delay(self.user_agent)

    async def close(self) -> None:
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()


class DataStorage:
    """Data storage manager for crawled content"""

    def __init__(self, storage_type: str = "json", base_path: str = "crawled_data"):
        """
        Initialize data storage

        Args:
            storage_type: Type of storage (json, csv, sqlite)
            base_path: Base path for file storage
        """
        self.storage_type = storage_type
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        # Create subdirectories
        (self.base_path / "json").mkdir(exist_ok=True)
        (self.base_path / "csv").mkdir(exist_ok=True)

        self.db_path = self.base_path / "crawl_data.db"

        logger.info(f"DataStorage initialized: {storage_type} storage at {base_path}")

    async def save_page_data(self, url: str, data: Dict[str, Any]) -> None:
        """
        Save page data

        Args:
            url: URL of the page
            data: Page data dictionary
        """
        try:
            if self.storage_type == "json":
                await self._save_json(url, data)
            elif self.storage_type == "csv":
                await self._save_csv(url, data)
            elif self.storage_type == "sqlite":
                await self._save_sqlite(url, data)
            else:
                logger.error(f"Unknown storage type: {self.storage_type}")

        except Exception as e:
            logger.error(f"Error saving data for {url}: {str(e)}")

    async def _save_json(self, url: str, data: Dict[str, Any]) -> None:
        """Save data as JSON file"""
        # Create filename from URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()
        filename = self.base_path / "json" / f"{url_hash}.json"

        async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2, ensure_ascii=False))

    async def _save_csv(self, url: str, data: Dict[str, Any]) -> None:
        """Save data as CSV (append to file)"""
        filename = self.base_path / "csv" / "pages.csv"

        # Prepare row data
        row = {
            'url': url,
            'title': data.get('title', ''),
            'text_length': len(data.get('text', '')),
            'links_count': len(data.get('links', [])),
            'images_count': len(data.get('images', [])),
            'word_count': len(data.get('text', '').split()),
            'crawl_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status_code': data.get('status_code', ''),
            'error': data.get('error', '')
        }

        # Check if file exists to determine if we need headers
        file_exists = filename.exists()

        async with aiofiles.open(filename, 'a', encoding='utf-8', newline='') as f:
            if not file_exists:
                # Write headers
                headers = ','.join(row.keys()) + '\n'
                await f.write(headers)

            # Write data row
            values = [str(v).replace(',', ';').replace('\n', ' ') for v in row.values()]
            await f.write(','.join(values) + '\n')

    async def _save_sqlite(self, url: str, data: Dict[str, Any]) -> None:
        """Save data to SQLite database"""
        async with aiosqlite.connect(self.db_path) as db:
            # Create table if not exists
            await db.execute('''
                CREATE TABLE IF NOT EXISTS pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    title TEXT,
                    text TEXT,
                    text_length INTEGER,
                    links_count INTEGER,
                    images_count INTEGER,
                    word_count INTEGER,
                    crawl_time TEXT,
                    status_code INTEGER,
                    error TEXT,
                    metadata TEXT
                )
            ''')

            # Prepare metadata
            metadata = json.dumps({
                'headings': data.get('headings', {}),
                'tables': data.get('tables', []),
                'lists': data.get('lists', {}),
                'images': data.get('images', []),
                'links': data.get('links', [])
            })

            # Insert or replace data
            await db.execute('''
                INSERT OR REPLACE INTO pages
                (url, title, text, text_length, links_count, images_count,
                 word_count, crawl_time, status_code, error, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                url,
                data.get('title', ''),
                data.get('text', ''),
                len(data.get('text', '')),
                len(data.get('links', [])),
                len(data.get('images', [])),
                len(data.get('text', '').split()),
                time.strftime('%Y-%m-%d %H:%M:%S'),
                data.get('status_code', ''),
                data.get('error', ''),
                metadata
            ))

            await db.commit()

    async def get_statistics(self) -> Dict[str, int]:
        """Get storage statistics"""
        if self.storage_type == "json":
            json_files = list((self.base_path / "json").glob("*.json"))
            return {'pages_stored': len(json_files)}

        elif self.storage_type == "csv":
            csv_file = self.base_path / "csv" / "pages.csv"
            if csv_file.exists():
                async with aiofiles.open(csv_file, 'r', encoding='utf-8') as f:
                    lines = await f.readlines()
                    return {'pages_stored': max(0, len(lines) - 1)}  # Subtract header
            return {'pages_stored': 0}

        elif self.storage_type == "sqlite":
            if self.db_path.exists():
                async with aiosqlite.connect(self.db_path) as db:
                    async with db.execute("SELECT COUNT(*) FROM pages") as cursor:
                        count = await cursor.fetchone()
                        return {'pages_stored': count[0] if count else 0}
            return {'pages_stored': 0}

        return {'pages_stored': 0}


class PoliteCrawler(AdvancedCrawler):
    """Complete crawler with rate limiting and politeness features"""

    def __init__(self, config: CrawlConfig):
        """
        Initialize polite crawler

        Args:
            config: Crawler configuration
        """
        # Initialize parent with basic settings
        super().__init__(
            max_concurrent=config.max_concurrent,
            max_per_domain_concurrent=config.max_per_domain_concurrent,
            max_depth=config.max_depth
        )

        self.config = config
        self.rate_limiter = RateLimiter(
            requests_per_second=config.requests_per_second,
            per_domain=True
        )
        self.robots_parser = RobotsParser(config.user_agent)
        self.data_storage = DataStorage(config.storage_type)

        # User agent rotation
        self.user_agents = [
            config.user_agent,
            "Mozilla/5.0 (compatible; PoliteCrawler/1.0)",
            "CrawlerBot/1.0 (+http://example.com/bot)"
        ] if config.rotate_user_agent else [config.user_agent]

        # Statistics
        self.robots_blocks = 0
        self.rate_limited_requests = 0

        logger.info(f"PoliteCrawler initialized with config: {asdict(config)}")

    async def _process_url_with_retry(self, task) -> Optional[Dict[str, Any]]:
        """
        Process URL with retry logic and politeness

        Args:
            task: URL task to process

        Returns:
            Parsed data or None
        """
        domain = urlparse(task.url).netloc.lower()

        # Check robots.txt
        if self.config.respect_robots:
            robots_data = await self.robots_parser.fetch_robots(task.url)

            if not self.robots_parser.can_fetch(task.url, robots_data):
                self.robots_blocks += 1
                logger.info(f"🚫 Blocked by robots.txt: {task.url}")
                return None

            # Apply crawl delay from robots.txt
            crawl_delay = self.robots_parser.get_crawl_delay(robots_data)
            if crawl_delay:
                await asyncio.sleep(max(crawl_delay, self.config.min_delay))

        # Apply rate limiting
        await self.rate_limiter.acquire(domain)

        # Add jitter for human-like behavior
        if self.config.jitter:
            jitter_time = random.uniform(0, self.config.min_delay * 0.3)
            await asyncio.sleep(jitter_time)

        # Process with retry logic
        last_error = None
        for attempt in range(self.config.retry_attempts + 1):
            try:
                # Update session headers with current user agent
                current_user_agent = random.choice(self.user_agents)
                if self.session:
                    self.session.headers.update({'User-Agent': current_user_agent})

                # Process the URL
                parsed_data = await self._process_url(task)

                if parsed_data:
                    # Convert to dict for storage
                    result_data = {
                        'url': parsed_data.url,
                        'title': parsed_data.title,
                        'text': parsed_data.text,
                        'links': parsed_data.links,
                        'images': parsed_data.images,
                        'headings': parsed_data.headings,
                        'tables': parsed_data.tables,
                        'lists': parsed_data.lists,
                        'metadata': parsed_data.metadata,
                        'status_code': 200,
                        'error': '',
                        'crawl_time': time.strftime('%Y-%m-%d %H:%M:%S')
                    }

                    # Save to storage
                    await self.data_storage.save_page_data(task.url, result_data)

                    return result_data

            except Exception as e:
                last_error = str(e)
                logger.warning(f"⚠️ Attempt {attempt + 1} failed for {task.url}: {last_error}")

                if attempt < self.config.retry_attempts:
                    # Exponential backoff
                    retry_delay = self.config.retry_delay * (2 ** attempt)
                    await asyncio.sleep(retry_delay)

        # All attempts failed
        error_data = {
            'url': task.url,
            'title': '',
            'text': '',
            'links': [],
            'images': [],
            'headings': {},
            'tables': [],
            'lists': {},
            'metadata': {},
            'status_code': 0,
            'error': last_error or 'Unknown error',
            'crawl_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        await self.data_storage.save_page_data(task.url, error_data)
        return None

    async def crawl_polite(self, start_urls: List[str], max_pages: int = 100) -> Dict[str, Any]:
        """
        Polite crawling with full feature set

        Args:
            start_urls: Starting URLs
            max_pages: Maximum pages to crawl

        Returns:
            Crawling results and statistics
        """
        logger.info(f"🚀 Starting polite crawl: {len(start_urls)} URLs, max_pages={max_pages}")

        # Initialize crawler state
        self.running = True
        self.stats = CrawlStats()
        self.results: Dict[str, Any] = {}

        # Set up base domains
        for url in start_urls:
            domain = urlparse(url).netloc.lower()
            self.base_domains.add(domain)

        # Configure URL filtering
        self.set_url_filtering(same_domain_only=self.config.same_domain_only)

        # Add start URLs to queue
        for url in start_urls:
            if self._should_crawl_url(url, 0):
                added = await self.queue.add_url(url, 0, 0)
                if added:
                    self.stats.urls_queued += 1

        # Create worker tasks
        worker_tasks = []
        for i in range(self.config.max_concurrent):
            task = asyncio.create_task(self._polite_worker())
            worker_tasks.append(task)

        # Create progress reporter task
        progress_task = asyncio.create_task(self._enhanced_progress_reporter())

        try:
            # Main crawling loop
            while (self.stats.urls_processed < max_pages and
                   (self.queue.get_stats()['queue_size'] > 0 or
                    await self.semaphore_manager.get_active_count() > 0)):

                await asyncio.sleep(0.1)

            # Stop workers
            self.running = False

            # Wait for workers to finish
            await asyncio.gather(*worker_tasks, return_exceptions=True)

            # Stop progress reporter
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

            # Get final statistics
            storage_stats = await self.data_storage.get_statistics()

            final_stats = {
                'crawl_stats': await self.get_crawl_statistics(),
                'storage_stats': storage_stats,
                'politeness_stats': {
                    'robots_blocks': self.robots_blocks,
                    'rate_limited_requests': self.rate_limited_requests
                }
            }

            logger.info(f"🎉 Polite crawl completed: {self.stats.urls_processed} pages processed")
            return final_stats

        except Exception as e:
            logger.error(f"❌ Polite crawling error: {str(e)}")
            self.running = False

            # Cancel all tasks
            for task in worker_tasks:
                task.cancel()
            progress_task.cancel()

            raise

        finally:
            await self.robots_parser.close()

    async def _polite_worker(self) -> None:
        """Worker for polite crawling"""
        while self.running:
            try:
                # Get next task (non-blocking)
                task = await self.queue.get_next_nowait()
                if task is None:
                    await asyncio.sleep(0.1)
                    continue

                # Process with politeness and retry
                result = await self._process_url_with_retry(task)
                if result:
                    self.stats.urls_processed += 1

            except Exception as e:
                logger.error(f"❌ Polite worker error: {str(e)}")
                await asyncio.sleep(0.1)

    async def _enhanced_progress_reporter(self, report_interval: float = 5.0) -> None:
        """Enhanced progress reporter with politeness stats"""
        while self.running:
            try:
                await asyncio.sleep(report_interval)

                queue_stats = self.queue.get_stats()
                active_count = await self.semaphore_manager.get_active_count()
                storage_stats = await self.data_storage.get_statistics()

                logger.info(
                    f"📊 Progress: {self.stats.urls_processed} processed, "
                    f"{queue_stats['queue_size']} queued, "
                    f"{active_count} active, "
                    f"{storage_stats['pages_stored']} stored, "
                    f"🚫 {self.robots_blocks} robots blocks, "
                    f"⚡ {self.stats.pages_per_second:.1f} pages/sec"
                )

            except Exception as e:
                logger.error(f"Progress reporter error: {str(e)}")


async def demo_polite_crawling():
    """Demonstrate complete polite crawler"""
    print("🤖 Polite Crawler Demo")
    print("=" * 40)

    # Configure crawler
    config = CrawlConfig(
        max_concurrent=3,
        max_per_domain_concurrent=1,
        requests_per_second=0.5,  # 1 request every 2 seconds
        min_delay=1.0,
        jitter=True,
        respect_robots=True,
        rotate_user_agent=False,
        retry_attempts=2,
        max_depth=2,
        storage_type="json"
    )

    crawler = PoliteCrawler(config)

    # Configure URL filtering
    crawler.set_url_filtering(
        same_domain_only=True,
        exclude_patterns=[
            r'.*\.(pdf|zip|exe|jpg|png|gif)$',
            r'.*/logout.*',
        ]
    )

    # Start crawling
    start_urls = [
        "https://example.com",
        "https://httpbin.org"
    ]

    try:
        stats = await crawler.crawl_polite(
            start_urls=start_urls,
            max_pages=10
        )

        # Display comprehensive statistics
        print(f"\n📊 Final Crawling Statistics:")
        crawl_stats = stats['crawl_stats']

        print(f"  ⏱️  Elapsed time: {crawl_stats['timing']['elapsed_time']:.1f}s")
        print(f"  📄 Pages processed: {crawl_stats['counts']['urls_processed']}")
        print(f"  ❌ Pages failed: {crawl_stats['counts']['urls_failed']}")
        print(f"  ⚡ Speed: {crawl_stats['timing']['pages_per_second']:.1f} pages/sec")
        print(f"  💾 Pages stored: {stats['storage_stats']['pages_stored']}")

        politeness_stats = stats['politeness_stats']
        print(f"  🤖 Robots blocks: {politeness_stats['robots_blocks']}")
        print(f"  ⏱️ Rate limited requests: {politeness_stats['rate_limited_requests']}")

        print(f"\n🎉 Polite crawling demo completed!")
        print(f"💾 Data saved to: crawled_data/ directory")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

    finally:
        await crawler.close()


async def main():
    """Main demonstration function"""
    try:
        await demo_polite_crawling()

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())