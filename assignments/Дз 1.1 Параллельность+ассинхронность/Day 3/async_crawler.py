"""
Day 1 - Basic Async HTTP Client
Implementation of AsyncCrawler class for parallel web page downloading
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('crawler.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of a single URL fetch operation"""
    url: str
    content: str
    status_code: int
    success: bool
    error_message: Optional[str] = None
    response_time: float = 0.0


class AsyncCrawler:
    """Asynchronous web crawler with concurrent URL fetching"""

    def __init__(self, max_concurrent: int = 10, timeout: int = 30):
        """
        Initialize the async crawler

        Args:
            max_concurrent: Maximum number of concurrent requests
            timeout: Request timeout in seconds
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(f"AsyncCrawler initialized with max_concurrent={max_concurrent}, timeout={timeout}s")

    async def _create_session(self) -> None:
        """Create aiohttp session with proper configuration"""
        if self.session is None or self.session.closed:
            timeout_config = aiohttp.ClientTimeout(
                connect=10,  # Connection timeout
                total=self.timeout,  # Total timeout
                sock_read=15  # Socket read timeout
            )

            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent + 2,  # Connection pool size
                limit_per_host=self.max_concurrent,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )

            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout_config,
                headers={
                    'User-Agent': 'AsyncCrawler/1.0 (Educational Purpose)'
                }
            )

    async def fetch_url(self, url: str) -> FetchResult:
        """
        Fetch a single URL asynchronously

        Args:
            url: URL to fetch

        Returns:
            FetchResult with content and metadata
        """
        await self._create_session()

        start_time = time.time()
        logger.info(f"🚀 Starting fetch for URL: {url}")

        async with self.semaphore:  # Limit concurrent requests
            try:
                async with self.session.get(url) as response:
                    response_time = time.time() - start_time

                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"✅ Successfully fetched {url} ({response.status}) in {response_time:.2f}s")
                        return FetchResult(
                            url=url,
                            content=content,
                            status_code=response.status,
                            success=True,
                            response_time=response_time
                        )
                    else:
                        error_msg = f"HTTP {response.status}: {response.reason}"
                        logger.warning(f"⚠️ Failed to fetch {url}: {error_msg} in {response_time:.2f}s")
                        return FetchResult(
                            url=url,
                            content="",
                            status_code=response.status,
                            success=False,
                            error_message=error_msg,
                            response_time=response_time
                        )

            except aiohttp.ClientConnectorError as e:
                response_time = time.time() - start_time
                error_msg = f"Connection error: {str(e)}"
                logger.error(f"❌ Connection error for {url}: {error_msg}")
                return FetchResult(
                    url=url,
                    content="",
                    status_code=0,
                    success=False,
                    error_message=error_msg,
                    response_time=response_time
                )

            except asyncio.TimeoutError:
                response_time = time.time() - start_time
                error_msg = f"Request timeout after {self.timeout}s"
                logger.error(f"⏰ Timeout for {url}: {error_msg}")
                return FetchResult(
                    url=url,
                    content="",
                    status_code=0,
                    success=False,
                    error_message=error_msg,
                    response_time=response_time
                )

            except aiohttp.ClientResponseError as e:
                response_time = time.time() - start_time
                error_msg = f"HTTP error: {str(e)}"
                logger.error(f"🚫 HTTP error for {url}: {error_msg}")
                return FetchResult(
                    url=url,
                    content="",
                    status_code=e.status,
                    success=False,
                    error_message=error_msg,
                    response_time=response_time
                )

            except Exception as e:
                response_time = time.time() - start_time
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"💥 Unexpected error for {url}: {error_msg}")
                return FetchResult(
                    url=url,
                    content="",
                    status_code=0,
                    success=False,
                    error_message=error_msg,
                    response_time=response_time
                )

    async def fetch_urls(self, urls: List[str]) -> Dict[str, FetchResult]:
        """
        Fetch multiple URLs concurrently

        Args:
            urls: List of URLs to fetch

        Returns:
            Dictionary mapping URLs to FetchResult objects
        """
        logger.info(f"📋 Starting to fetch {len(urls)} URLs with max_concurrent={self.max_concurrent}")

        # Create tasks for all URLs
        tasks = [self.fetch_url(url) for url in urls]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and create dictionary
        url_results = {}
        successful = 0
        failed = 0

        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                # Handle unexpected exceptions
                url_results[url] = FetchResult(
                    url=url,
                    content="",
                    status_code=0,
                    success=False,
                    error_message=f"Task exception: {str(result)}"
                )
                failed += 1
            else:
                url_results[url] = result
                if result.success:
                    successful += 1
                else:
                    failed += 1

        logger.info(f"📊 Fetch completed: {successful} successful, {failed} failed")
        return url_results

    async def fetch_urls_sequential(self, urls: List[str]) -> Dict[str, FetchResult]:
        """
        Fetch URLs sequentially (for performance comparison)

        Args:
            urls: List of URLs to fetch

        Returns:
            Dictionary mapping URLs to FetchResult objects
        """
        logger.info(f"🐌 Starting sequential fetch of {len(urls)} URLs")

        url_results = {}
        for url in urls:
            result = await self.fetch_url(url)
            url_results[url] = result

        return url_results

    async def close(self) -> None:
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("🔒 AsyncCrawler session closed")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


async def demo_performance_comparison():
    """Demonstrate performance comparison between parallel and sequential fetching"""
    print("🚀 AsyncCrawler Performance Demo")
    print("=" * 50)

    # Test URLs with different response times
    test_urls = [
        "https://example.com",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/json",
        "https://jsonplaceholder.typicode.com/posts/1",
        "https://httpbin.org/uuid",
        "https://httpbin.org/delay/3",
        "https://jsonplaceholder.typicode.com/users/1"
    ]

    # Parallel fetch
    print("\n📡 Parallel fetching...")
    start_time = time.time()

    async with AsyncCrawler(max_concurrent=5) as crawler:
        parallel_results = await crawler.fetch_urls(test_urls)

    parallel_time = time.time() - start_time

    # Sequential fetch
    print("\n🐌 Sequential fetching...")
    start_time = time.time()

    async with AsyncCrawler(max_concurrent=1) as crawler:
        sequential_results = await crawler.fetch_urls_sequential(test_urls)

    sequential_time = time.time() - start_time

    # Calculate statistics
    parallel_success = sum(1 for r in parallel_results.values() if r.success)
    sequential_success = sum(1 for r in sequential_results.values() if r.success)

    # Print results
    print(f"\n📊 Performance Comparison Results:")
    print(f"Parallel fetch:   {parallel_time:.2f}s ({parallel_success}/{len(test_urls)} successful)")
    print(f"Sequential fetch: {sequential_time:.2f}s ({sequential_success}/{len(test_urls)} successful)")
    print(f"Speedup: {sequential_time/parallel_time:.1f}x faster")

    # Print individual results
    print(f"\n📋 Individual Results:")
    for url, result in parallel_results.items():
        status = "✅" if result.success else "❌"
        time_info = f"({result.response_time:.2f}s)"
        print(f"{status} {time_info} {url}")
        if not result.success:
            print(f"   Error: {result.error_message}")


async def test_error_handling():
    """Test error handling capabilities"""
    print("\n🛡️ Error Handling Tests")
    print("=" * 30)

    # Test URLs that will cause different types of errors
    error_test_urls = [
        "https://example.com",  # Should work
        "https://httpbin.org/status/404",  # 404 error
        "https://httpbin.org/status/500",  # 500 error
        "https://nonexistent-domain-12345.com",  # Connection error
        "https://httpbin.org/delay/10",  # Might timeout
    ]

    async with AsyncCrawler(max_concurrent=3, timeout=5) as crawler:
        results = await crawler.fetch_urls(error_test_urls)

        print(f"\n🔍 Error handling results:")
        for url, result in results.items():
            if result.success:
                print(f"✅ {url} - Success ({result.status_code})")
            else:
                print(f"❌ {url} - {result.error_message}")


async def main():
    """Main demonstration function"""
    try:
        await demo_performance_comparison()
        await test_error_handling()
        print("\n🎉 Demo completed successfully!")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())