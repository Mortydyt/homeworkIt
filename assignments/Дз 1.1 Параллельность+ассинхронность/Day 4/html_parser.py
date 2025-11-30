"""
Day 2 - HTML Parsing and Data Extraction
Implementation of HTMLParser class with BeautifulSoup integration
"""

import asyncio
import aiohttp
import json
import re
import logging
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse, urlunparse
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup, Tag
from pathlib import Path

# Import our previous crawler
import sys
sys.path.append(str(Path(__file__).parent / "Day 1"))
from async_crawler import AsyncCrawler, FetchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('html_parser.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class ParsedData:
    """Data structure for parsed HTML content"""
    url: str
    title: str
    text: str
    text_length: int
    links: List[str]
    links_count: int
    images: List[Dict[str, str]]
    images_count: int
    headings: Dict[str, List[str]]
    tables: List[List[Dict[str, Union[str, List[str]]]]]
    lists: Dict[str, List[str]]
    metadata: Dict[str, str]
    word_count: int
    fetch_time: float
    parse_time: float


class HTMLParser:
    """HTML content parser with structured data extraction"""

    def __init__(self):
        """Initialize the HTML parser"""
        self.stats = {
            'pages_parsed': 0,
            'links_extracted': 0,
            'images_extracted': 0,
            'parse_errors': 0
        }
        logger.info("HTMLParser initialized")

    async def parse_html(self, html: str, url: str) -> Optional[ParsedData]:
        """
        Parse HTML content and extract structured data

        Args:
            html: HTML content string
            url: Source URL for resolving relative links

        Returns:
            ParsedData object with extracted information
        """
        start_time = asyncio.get_event_loop().time()

        try:
            soup = BeautifulSoup(html, 'lxml')
            parse_time = asyncio.get_event_loop().time() - start_time

            # Extract all data
            title = self._extract_title(soup)
            text = self._extract_text(soup)
            links = self._extract_links(soup, url)
            images = self._extract_images(soup, url)
            headings = self._extract_headings(soup)
            tables = self._extract_tables(soup)
            lists_data = self._extract_lists(soup)
            metadata = self._extract_metadata(soup)

            # Create parsed data object
            parsed_data = ParsedData(
                url=url,
                title=title,
                text=text,
                text_length=len(text),
                links=links,
                links_count=len(links),
                images=images,
                images_count=len(images),
                headings=headings,
                tables=tables,
                lists=lists_data,
                metadata=metadata,
                word_count=len(text.split()),
                fetch_time=0.0,  # Will be set by caller
                parse_time=parse_time
            )

            # Update statistics
            self.stats['pages_parsed'] += 1
            self.stats['links_extracted'] += len(links)
            self.stats['images_extracted'] += len(images)

            logger.info(f"✅ Successfully parsed {url} - {len(links)} links, {len(images)} images")
            return parsed_data

        except Exception as e:
            self.stats['parse_errors'] += 1
            logger.error(f"❌ Error parsing {url}: {str(e)}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else ""

    def _extract_text(self, soup: BeautifulSoup, selector: Optional[str] = None) -> str:
        """
        Extract clean text from HTML

        Args:
            soup: BeautifulSoup object
            selector: CSS selector to focus extraction

        Returns:
            Clean text content
        """
        if selector:
            elements = soup.select(selector)
            if elements:
                soup = BeautifulSoup(''.join(str(elem) for elem in elements), 'lxml')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract all links from page and convert to absolute URLs

        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links

        Returns:
            List of absolute URLs
        """
        links = set()
        base_domain = urlparse(base_url).netloc

        for link in soup.find_all('a', href=True):
            href = link['href'].strip()

            # Skip empty links, anchors, and javascript
            if (not href or href.startswith('#') or
                href.startswith('javascript:') or href.startswith('mailto:')):
                continue

            # Convert to absolute URL
            try:
                absolute_url = urljoin(base_url, href)

                # Validate and normalize URL
                parsed = urlparse(absolute_url)
                if parsed.scheme in ('http', 'https') and parsed.netloc:
                    # Remove fragment
                    clean_url = urlunparse((
                        parsed.scheme,
                        parsed.netloc,
                        parsed.path,
                        parsed.params,
                        parsed.query,
                        ''  # Remove fragment
                    ))
                    links.add(clean_url)
            except Exception as e:
                logger.warning(f"⚠️ Error processing link {href}: {str(e)}")
                continue

        return sorted(list(links))

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """
        Extract all images with src and alt attributes

        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative URLs

        Returns:
            List of image information dictionaries
        """
        images = []

        for img in soup.find_all('img'):
            img_info = {
                'src': '',
                'alt': '',
                'title': '',
                'width': '',
                'height': ''
            }

            # Extract attributes
            if img.get('src'):
                try:
                    img_info['src'] = urljoin(base_url, img['src'])
                except Exception:
                    img_info['src'] = img['src']

            img_info['alt'] = img.get('alt', '')
            img_info['title'] = img.get('title', '')
            img_info['width'] = img.get('width', '')
            img_info['height'] = img.get('height', '')

            # Skip images without src
            if img_info['src']:
                images.append(img_info)

        return images

    def _extract_headings(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """
        Extract all headings (h1, h2, h3)

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with heading levels as keys
        """
        headings = {
            'h1': [],
            'h2': [],
            'h3': []
        }

        for level in headings:
            for heading in soup.find_all(level):
                text = heading.get_text().strip()
                if text:
                    headings[level].append(text)

        return headings

    def _extract_tables(self, soup: BeautifulSoup) -> List[List[Dict[str, Union[str, List[str]]]]]:
        """
        Extract table data

        Args:
            soup: BeautifulSoup object

        Returns:
            List of tables with row data
        """
        tables_data = []

        for table in soup.find_all('table'):
            table_data = []

            # Extract headers
            headers = []
            header_row = table.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td']):
                    headers.append(th.get_text().strip())

            # Extract data rows
            for row in table.find_all('tr')[1:]:  # Skip header row
                row_data = {}
                cells = row.find_all(['td', 'th'])

                for i, cell in enumerate(cells):
                    header = headers[i] if i < len(headers) else f'Column_{i+1}'
                    row_data[header] = cell.get_text().strip()

                if row_data:
                    table_data.append(row_data)

            if table_data:
                tables_data.append(table_data)

        return tables_data

    def _extract_lists(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """
        Extract list items (ul, ol)

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with list types as keys
        """
        lists_data = {
            'unordered': [],
            'ordered': [],
            'definition': []
        }

        # Unordered lists
        for ul in soup.find_all('ul'):
            for li in ul.find_all('li', recursive=False):
                text = li.get_text().strip()
                if text:
                    lists_data['unordered'].append(text)

        # Ordered lists
        for ol in soup.find_all('ol'):
            for li in ol.find_all('li', recursive=False):
                text = li.get_text().strip()
                if text:
                    lists_data['ordered'].append(text)

        # Definition lists
        for dl in soup.find_all('dl'):
            for dt in dl.find_all('dt'):
                term = dt.get_text().strip()
                dd = dt.find_next_sibling('dd')
                definition = dd.get_text().strip() if dd else ""
                if term:
                    lists_data['definition'].append(f"{term}: {definition}")

        return lists_data

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """
        Extract metadata from meta tags

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with metadata
        """
        metadata = {}

        # Extract specific meta tags
        meta_tags = {
            'description': ['name="description"', 'property="og:description"'],
            'keywords': ['name="keywords"'],
            'author': ['name="author"'],
            'viewport': ['name="viewport"'],
            'robots': ['name="robots"'],
            'title': ['property="og:title"'],
            'site_name': ['property="og:site_name"'],
            'type': ['property="og:type"'],
            'image': ['property="og:image"']
        }

        for meta_name, selectors in meta_tags.items():
            for selector in selectors:
                meta_tag = soup.find('meta', attrs={'name': selector.split('=')[1].strip('"')})
                if not meta_tag:
                    meta_tag = soup.find('meta', attrs={'property': selector.split('=')[1].strip('"')})

                if meta_tag and meta_tag.get('content'):
                    metadata[meta_name] = meta_tag['content']
                    break

        return metadata

    def get_statistics(self) -> Dict[str, int]:
        """Get parsing statistics"""
        return self.stats.copy()


class AsyncCrawlerWithParser(AsyncCrawler):
    """Extended AsyncCrawler with HTML parsing capabilities"""

    def __init__(self, max_concurrent: int = 10, timeout: int = 30):
        super().__init__(max_concurrent, timeout)
        self.parser = HTMLParser()

    async def fetch_and_parse(self, url: str) -> Optional[ParsedData]:
        """
        Fetch URL and parse its content

        Args:
            url: URL to fetch and parse

        Returns:
            ParsedData object or None if failed
        """
        fetch_start = asyncio.get_event_loop().time()

        # Fetch the URL
        result = await self.fetch_url(url)

        if not result.success:
            logger.error(f"❌ Failed to fetch {url}: {result.error_message}")
            return None

        # Parse the content
        parsed_data = await self.parser.parse_html(result.content, url)

        if parsed_data:
            parsed_data.fetch_time = result.response_time

        return parsed_data

    async def fetch_and_parse_urls(self, urls: List[str]) -> Dict[str, Optional[ParsedData]]:
        """
        Fetch and parse multiple URLs concurrently

        Args:
            urls: List of URLs to fetch and parse

        Returns:
            Dictionary mapping URLs to ParsedData objects (or None if failed)
        """
        logger.info(f"📋 Starting to fetch and parse {len(urls)} URLs")

        # Create tasks for all URLs
        tasks = [self.fetch_and_parse(url) for url in urls]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and create dictionary
        url_results = {}
        successful = 0
        failed = 0

        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                # Handle unexpected exceptions
                logger.error(f"💥 Unexpected error for {url}: {str(result)}")
                url_results[url] = None
                failed += 1
            else:
                url_results[url] = result
                if result is not None:
                    successful += 1
                else:
                    failed += 1

        logger.info(f"📊 Fetch and parse completed: {successful} successful, {failed} failed")
        return url_results

    def get_parser_statistics(self) -> Dict[str, int]:
        """Get parser statistics"""
        return self.parser.get_statistics()


async def demo_html_parsing():
    """Demonstrate HTML parsing functionality"""
    print("📄 HTML Parser Demo")
    print("=" * 40)

    # Test URLs with different content types
    test_urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://jsonplaceholder.typicode.com",
        "https://httpbin.org/links/5",
        "https://www.python.org"
    ]

    async with AsyncCrawlerWithParser(max_concurrent=3) as crawler:
        results = await crawler.fetch_and_parse_urls(test_urls)

        # Process and display results
        for url, parsed_data in results.items():
            print(f"\n🔍 Results for: {url}")
            if parsed_data:
                print(f"  📝 Title: {parsed_data.title}")
                print(f"  📊 Text length: {parsed_data.text_length} characters")
                print(f"  🔗 Links: {parsed_data.links_count}")
                print(f"  🖼️ Images: {parsed_data.images_count}")
                print(f"  📖 Words: {parsed_data.word_count}")
                print(f"  ⏱️ Fetch time: {parsed_data.fetch_time:.2f}s")
                print(f"  ⚡ Parse time: {parsed_data.parse_time:.2f}s")

                # Show first few links
                if parsed_data.links:
                    print(f"  🔗 First 3 links: {parsed_data.links[:3]}")

                # Show metadata
                if parsed_data.metadata:
                    print(f"  📋 Metadata: {parsed_data.metadata}")
            else:
                print("  ❌ Failed to fetch or parse")

        # Show overall statistics
        parser_stats = crawler.get_parser_statistics()
        print(f"\n📊 Overall Statistics:")
        print(f"  📄 Pages parsed: {parser_stats['pages_parsed']}")
        print(f"  🔗 Links extracted: {parser_stats['links_extracted']}")
        print(f"  🖼️ Images extracted: {parser_stats['images_extracted']}")
        print(f"  ❌ Parse errors: {parser_stats['parse_errors']}")


async def save_results_to_json(results: Dict[str, Optional[ParsedData]], filename: str = "parsed_results.json"):
    """Save parsing results to JSON file"""
    # Convert results to serializable format
    serializable_results = {}

    for url, parsed_data in results.items():
        if parsed_data:
            serializable_results[url] = asdict(parsed_data)
        else:
            serializable_results[url] = None

    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to {filename}")


async def main():
    """Main demonstration function"""
    try:
        await demo_html_parsing()

        # Additional demo with saving results
        print(f"\n💾 Saving results demonstration...")
        test_urls = ["https://example.com", "https://httpbin.org/html"]

        async with AsyncCrawlerWithParser(max_concurrent=2) as crawler:
            results = await crawler.fetch_and_parse_urls(test_urls)
            await save_results_to_json(results, "demo_parsed_results.json")

        print("\n🎉 HTML parsing demo completed successfully!")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())