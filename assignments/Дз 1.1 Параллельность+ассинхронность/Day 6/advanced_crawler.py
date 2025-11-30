"""
Day 3 - Concurrency Management and Queues
Implementation of advanced crawler with queue management and concurrency control
"""

import asyncio
import aiohttp
import time
import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field
from collections import defaultdict, deque
from priority_queue import PriorityQueue
import heapq

# Import previous modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "Day 2"))
from html_parser import AsyncCrawlerWithParser, ParsedData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('advanced_crawler.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class URLTask:
    """Represents a URL to be crawled"""
    url: str
    depth: int
    priority: int = 0
    source_url: Optional[str] = None
    retry_count: int = 0

    def __lt__(self, other):
        """For priority queue comparison (higher priority = lower number)"""
        return self.priority < other.priority


@dataclass
class CrawlStats:
    """Crawling statistics"""
    start_time: float = field(default_factory=time.time)
    urls_queued: int = 0
    urls_processed: int = 0
    urls_failed: int = 0
    urls_skipped: int = 0
    total_bytes_downloaded: int = 0
    active_tasks: int = 0
    domains_discovered: Set[str] = field(default_factory=set)
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def elapsed_time(self) -> float:
        """Time elapsed since start"""
        return time.time() - self.start_time

    @property
    def pages_per_second(self) -> float:
        """Current crawling speed"""
        if self.elapsed_time > 0:
            return self.urls_processed / self.elapsed_time
        return 0.0


class CrawlerQueue:
    """Priority queue for managing URLs to crawl"""

    def __init__(self, max_size: int = 10000):
        """
        Initialize crawler queue

        Args:
            max_size: Maximum number of URLs in queue
        """
        self.max_size = max_size
        self._queue: List[URLTask] = []
        self._queue_lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._queue_lock)

        # Tracking sets and dictionaries
        self.queued_urls: Set[str] = set()
        self.processed_urls: Set[str] = set()
        self.failed_urls: Dict[str, str] = {}  # url -> error_message
        self.url_depth: Dict[str, int] = {}  # url -> depth

        logger.info(f"CrawlerQueue initialized with max_size={max_size}")

    async def add_url(self, url: str, depth: int, priority: int = 0, source_url: Optional[str] = None) -> bool:
        """
        Add URL to queue if not already processed or queued

        Args:
            url: URL to add
            depth: Current depth
            priority: Priority (lower = higher priority)
            source_url: Source URL where this was found

        Returns:
            True if URL was added, False if already exists
        """
        async with self._queue_lock:
            # Check if URL already processed or queued
            if url in self.processed_urls or url in self.queued_urls:
                return False

            # Check queue size limit
            if len(self._queue) >= self.max_size:
                logger.warning(f"Queue is full (max_size={self.max_size}), skipping URL: {url}")
                return False

            # Create and add task
            task = URLTask(url=url, depth=depth, priority=priority, source_url=source_url)
            heapq.heappush(self._queue, task)
            self.queued_urls.add(url)
            self.url_depth[url] = depth

            # Notify waiting consumers
            self._not_empty.notify()

            return True

    async def get_next(self) -> Optional[URLTask]:
        """
        Get next URL from queue (blocks if empty)

        Returns:
            URLTask or None if queue is shutting down
        """
        async with self._not_empty:
            # Wait for queue to have items
            while not self._queue:
                await self._not_empty.wait()

            # Get highest priority item (lowest priority number)
            task = heapq.heappop(self._queue)
            self.queued_urls.discard(task.url)

            return task

    async def get_next_nowait(self) -> Optional[URLTask]:
        """
        Get next URL from queue without blocking

        Returns:
            URLTask or None if queue is empty
        """
        async with self._queue_lock:
            if not self._queue:
                return None

            task = heapq.heappop(self._queue)
            self.queued_urls.discard(task.url)

            return task

    async def mark_processed(self, url: str) -> None:
        """Mark URL as successfully processed"""
        async with self._queue_lock:
            self.processed_urls.add(url)

    async def mark_failed(self, url: str, error: str) -> None:
        """Mark URL as failed"""
        async with self._queue_lock:
            self.processed_urls.add(url)
            self.failed_urls[url] = error

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        return {
            'queue_size': len(self._queue),
            'queued_urls': len(self.queued_urls),
            'processed_urls': len(self.processed_urls),
            'failed_urls': len(self.failed_urls)
        }

    def is_url_processed(self, url: str) -> bool:
        """Check if URL has been processed"""
        return url in self.processed_urls

    def is_url_queued(self, url: str) -> bool:
        """Check if URL is currently queued"""
        return url in self.queued_urls

    def get_depth(self, url: str) -> Optional[int]:
        """Get depth for a URL"""
        return self.url_depth.get(url)


class SemaphoreManager:
    """Manages semaphores for concurrency control"""

    def __init__(self, max_global_concurrent: int = 10, max_per_domain_concurrent: int = 3):
        """
        Initialize semaphore manager

        Args:
            max_global_concurrent: Global concurrency limit
            max_per_domain_concurrent: Per-domain concurrency limit
        """
        self.max_global_concurrent = max_global_concurrent
        self.max_per_domain_concurrent = max_per_domain_concurrent

        # Global semaphore
        self.global_semaphore = asyncio.Semaphore(max_global_concurrent)

        # Per-domain semaphores
        self.domain_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.semaphore_lock = asyncio.Lock()

        # Active tasks tracking
        self.active_tasks: Set[str] = set()
        self.active_tasks_lock = asyncio.Lock()

        logger.info(f"SemaphoreManager initialized: global={max_global_concurrent}, per_domain={max_per_domain_concurrent}")

    async def acquire(self, url: str) -> Tuple[asyncio.Semaphore, str]:
        """
        Acquire appropriate semaphores for URL

        Args:
            url: URL to crawl

        Returns:
            Tuple of (semaphore, domain)
        """
        domain = urlparse(url).netloc.lower()

        # Get or create domain semaphore
        async with self.semaphore_lock:
            if domain not in self.domain_semaphores:
                self.domain_semaphores[domain] = asyncio.Semaphore(self.max_per_domain_concurrent)
            domain_semaphore = self.domain_semaphores[domain]

        # Add to active tasks
        async with self.active_tasks_lock:
            self.active_tasks.add(url)

        # Acquire semaphores (global first, then domain)
        await self.global_semaphore.acquire()
        await domain_semaphore.acquire()

        return domain_semaphore, domain

    async def release(self, url: str, domain_semaphore: asyncio.Semaphore, domain: str) -> None:
        """
        Release semaphores for URL

        Args:
            url: URL that was processed
            domain_semaphore: Domain semaphore to release
            domain: Domain name
        """
        # Remove from active tasks
        async with self.active_tasks_lock:
            self.active_tasks.discard(url)

        # Release semaphores
        domain_semaphore.release()
        self.global_semaphore.release()

    async def get_active_count(self) -> int:
        """Get number of active tasks"""
        async with self.active_tasks_lock:
            return len(self.active_tasks)

    def get_domain_stats(self) -> Dict[str, int]:
        """Get per-domain statistics"""
        return {
            domain: self.max_per_domain_concurrent - sem._value
            for domain, sem in self.domain_semaphores.items()
        }


class AdvancedCrawler(AsyncCrawlerWithParser):
    """Advanced crawler with queue management and concurrency control"""

    def __init__(self, max_concurrent: int = 10, max_per_domain_concurrent: int = 3,
                 max_queue_size: int = 10000, max_depth: int = 3):
        """
        Initialize advanced crawler

        Args:
            max_concurrent: Global concurrency limit
            max_per_domain_concurrent: Per-domain concurrency limit
            max_queue_size: Maximum queue size
            max_depth: Maximum crawling depth
        """
        super().__init__(max_concurrent)

        self.max_depth = max_depth
        self.queue = CrawlerQueue(max_queue_size)
        self.semaphore_manager = SemaphoreManager(max_concurrent, max_per_domain_concurrent)
        self.stats = CrawlStats()

        # URL filtering
        self.same_domain_only: bool = True
        self.allowed_domains: Set[str] = set()
        self.exclude_patterns: List[str] = []
        self.include_patterns: List[str] = []

        # Crawler state
        self.base_domains: Set[str] = set()
        self.running: bool = False

        logger.info(f"AdvancedCrawler initialized: concurrent={max_concurrent}, max_depth={max_depth}")

    def set_url_filtering(self, same_domain_only: bool = True,
                         allowed_domains: Optional[List[str]] = None,
                         exclude_patterns: Optional[List[str]] = None,
                         include_patterns: Optional[List[str]] = None) -> None:
        """
        Configure URL filtering

        Args:
            same_domain_only: Only crawl same domains as start URLs
            allowed_domains: List of allowed domains
            exclude_patterns: Regex patterns to exclude
            include_patterns: Regex patterns to include (if set, others are excluded)
        """
        self.same_domain_only = same_domain_only
        self.allowed_domains = set(allowed_domains or [])
        self.exclude_patterns = exclude_patterns or []
        self.include_patterns = include_patterns or []

        logger.info(f"URL filtering configured: same_domain={same_domain_only}, "
                   f"allowed={len(self.allowed_domains)} domains, "
                   f"exclude={len(self.exclude_patterns)} patterns")

    def _should_crawl_url(self, url: str, depth: int) -> bool:
        """
        Check if URL should be crawled based on filtering rules

        Args:
            url: URL to check
            depth: Current depth

        Returns:
            True if URL should be crawled
        """
        # Check depth limit
        if depth > self.max_depth:
            return False

        # Check if already processed or queued
        if self.queue.is_url_processed(url) or self.queue.is_url_queued(url):
            return False

        # Parse domain
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Check same domain restriction
        if self.same_domain_only and self.base_domains and domain not in self.base_domains:
            return False

        # Check allowed domains
        if self.allowed_domains and domain not in self.allowed_domains:
            return False

        # Apply include patterns (if specified)
        if self.include_patterns:
            if not any(re.search(pattern, url) for pattern in self.include_patterns):
                return False

        # Apply exclude patterns
        if any(re.search(pattern, url) for pattern in self.exclude_patterns):
            return False

        # Skip non-HTTP URLs
        if parsed.scheme not in ('http', 'https'):
            return False

        return True

    async def _process_url(self, task: URLTask) -> Optional[ParsedData]:
        """
        Process a single URL task

        Args:
            task: URL task to process

        Returns:
            ParsedData or None if failed
        """
        # Acquire semaphores
        domain_semaphore, domain = await self.semaphore_manager.acquire(task.url)

        try:
            logger.debug(f"Processing {task.url} (depth: {task.depth})")

            # Fetch and parse URL
            parsed_data = await self.fetch_and_parse(task.url)

            if parsed_data:
                # Update statistics
                self.stats.urls_processed += 1
                self.stats.domains_discovered.add(domain)

                # Extract and queue new links if within depth limit
                if task.depth < self.max_depth:
                    await self._enqueue_links(parsed_data.links, task.url, task.depth + 1)

                # Mark as processed
                await self.queue.mark_processed(task.url)

                logger.debug(f"Successfully processed {task.url}")
            else:
                # Mark as failed
                await self.queue.mark_failed(task.url, "Failed to fetch or parse")
                self.stats.urls_failed += 1
                self.stats.errors_by_type["fetch_failed"] += 1

            return parsed_data

        except Exception as e:
            error_msg = str(e)
            await self.queue.mark_failed(task.url, error_msg)
            self.stats.urls_failed += 1

            # Categorize error
            if "timeout" in error_msg.lower():
                self.stats.errors_by_type["timeout"] += 1
            elif "connection" in error_msg.lower():
                self.stats.errors_by_type["connection"] += 1
            else:
                self.stats.errors_by_type["other"] += 1

            logger.error(f"Error processing {task.url}: {error_msg}")
            return None

        finally:
            # Release semaphores
            await self.semaphore_manager.release(task.url, domain_semaphore, domain)

    async def _enqueue_links(self, links: List[str], source_url: str, depth: int) -> None:
        """
        Enqueue new links found on a page

        Args:
            links: List of links to enqueue
            source_url: URL where links were found
            depth: Depth for new links
        """
        for link in links:
            if self._should_crawl_url(link, depth):
                # Calculate priority based on depth and other factors
                priority = depth  # Prefer shallower links

                # Add to queue
                added = await self.queue.add_url(link, depth, priority, source_url)
                if added:
                    self.stats.urls_queued += 1

    async def _worker(self) -> None:
        """Worker coroutine for processing URLs"""
        while self.running:
            try:
                # Get next task (non-blocking)
                task = await self.queue.get_next_nowait()
                if task is None:
                    # No tasks available, wait a bit
                    await asyncio.sleep(0.1)
                    continue

                # Process the task
                await self._process_url(task)

            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                await asyncio.sleep(0.1)

    async def _progress_reporter(self, report_interval: float = 5.0) -> None:
        """Progress reporting coroutine"""
        while self.running:
            try:
                await asyncio.sleep(report_interval)

                queue_stats = self.queue.get_stats()
                active_count = await self.semaphore_manager.get_active_count()

                logger.info(
                    f"📊 Progress: {self.stats.urls_processed} processed, "
                    f"{queue_stats['queue_size']} queued, "
                    f"{active_count} active, "
                    f"{self.stats.pages_per_second:.1f} pages/sec, "
                    f"elapsed: {self.stats.elapsed_time:.1f}s"
                )

            except Exception as e:
                logger.error(f"Progress reporter error: {str(e)}")

    async def crawl(self, start_urls: List[str], max_pages: int = 100,
                   same_domain_only: bool = True) -> Dict[str, ParsedData]:
        """
        Main crawling method

        Args:
            start_urls: Starting URLs
            max_pages: Maximum number of pages to crawl
            same_domain_only: Only crawl same domains

        Returns:
            Dictionary of URL -> ParsedData
        """
        logger.info(f"Starting crawl: {len(start_urls)} start URLs, max_pages={max_pages}")

        # Initialize crawler state
        self.running = True
        self.stats = CrawlStats()
        self.results: Dict[str, ParsedData] = {}

        # Set up base domains
        for url in start_urls:
            domain = urlparse(url).netloc.lower()
            self.base_domains.add(domain)

        # Configure URL filtering
        self.set_url_filtering(same_domain_only=same_domain_only)

        # Add start URLs to queue
        for url in start_urls:
            if self._should_crawl_url(url, 0):
                added = await self.queue.add_url(url, 0, 0)
                if added:
                    self.stats.urls_queued += 1

        # Create worker tasks
        worker_tasks = []
        for i in range(self.max_concurrent):
            task = asyncio.create_task(self._worker())
            worker_tasks.append(task)

        # Create progress reporter task
        progress_task = asyncio.create_task(self._progress_reporter())

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

            # Collect results
            # Note: In a real implementation, you'd store results during processing
            # For now, we'll return an empty dict

            logger.info(f"Crawling completed: {self.stats.urls_processed} pages processed")

            return self.results

        except Exception as e:
            logger.error(f"Crawling error: {str(e)}")
            self.running = False

            # Cancel all tasks
            for task in worker_tasks:
                task.cancel()
            progress_task.cancel()

            raise

    async def get_crawl_statistics(self) -> Dict[str, any]:
        """Get comprehensive crawling statistics"""
        queue_stats = self.queue.get_stats()
        domain_stats = self.semaphore_manager.get_domain_stats()
        active_count = await self.semaphore_manager.get_active_count()

        return {
            'timing': {
                'elapsed_time': self.stats.elapsed_time,
                'pages_per_second': self.stats.pages_per_second
            },
            'counts': {
                'urls_processed': self.stats.urls_processed,
                'urls_failed': self.stats.urls_failed,
                'urls_skipped': self.stats.urls_skipped,
                'urls_queued': self.stats.urls_queued,
                'active_tasks': active_count
            },
            'queue': queue_stats,
            'domains': {
                'discovered': len(self.stats.domains_discovered),
                'domains': list(self.stats.domains_discovered),
                'domain_activity': domain_stats
            },
            'errors': dict(self.stats.errors_by_type)
        }


async def demo_advanced_crawling():
    """Demonstrate advanced crawling functionality"""
    print("🚀 Advanced Crawler Demo")
    print("=" * 40)

    # Configure crawler
    crawler = AdvancedCrawler(
        max_concurrent=5,
        max_per_domain_concurrent=2,
        max_depth=2,
        max_queue_size=100
    )

    # Set up URL filtering
    crawler.set_url_filtering(
        same_domain_only=True,
        exclude_patterns=[
            r'.*\.(pdf|zip|exe|jpg|png|gif)$',  # Skip files
            r'.*/logout.*',                     # Skip logout links
            r'.*\?.*(sort|filter|search)=.*'    # Skip dynamic sorting/filtering
        ]
    )

    # Start crawling
    start_urls = [
        "https://example.com",
        "https://httpbin.org"
    ]

    try:
        results = await crawler.crawl(
            start_urls=start_urls,
            max_pages=20,
            same_domain_only=True
        )

        # Display final statistics
        stats = await crawler.get_crawl_statistics()

        print(f"\n📊 Final Statistics:")
        print(f"  ⏱️  Elapsed time: {stats['timing']['elapsed_time']:.1f}s")
        print(f"  📄 Pages processed: {stats['counts']['urls_processed']}")
        print(f"  ❌ Pages failed: {stats['counts']['urls_failed']}")
        print(f"  ⚡ Speed: {stats['timing']['pages_per_second']:.1f} pages/sec")
        print(f"  🌐 Domains discovered: {stats['domains']['discovered']}")

        if stats['errors']:
            print(f"  🚨 Errors: {stats['errors']}")

        print(f"\n🎉 Crawling demo completed!")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

    finally:
        await crawler.close()


async def main():
    """Main demonstration function"""
    try:
        await demo_advanced_crawling()

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())