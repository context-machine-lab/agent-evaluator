"""Default tools for agents based on DeepResearchAgent patterns."""

import asyncio
import subprocess
import json
import os
import sys
import tempfile
import traceback
import re
from typing import Any, Dict, Optional
from pathlib import Path

# For web tools
try:
    import requests
    from bs4 import BeautifulSoup
    WEB_TOOLS_AVAILABLE = True
except ImportError:
    WEB_TOOLS_AVAILABLE = False

# For data processing
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from ..core.registry import register_tool, TOOL


@register_tool("python_interpreter")
async def python_interpreter_tool(code: str, timeout: int = 30, **kwargs) -> str:
    """Execute Python code and return the output.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Execution output or error message
    """
    try:
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Add imports that might be commonly needed for data tasks
            setup_code = """
import sys
import json
import math
import re
from collections import defaultdict, Counter
import datetime

# Capture print outputs
__output__ = []
original_print = print

def captured_print(*args, **kwargs):
    output = ' '.join(str(arg) for arg in args)
    __output__.append(output)
    original_print(*args, **kwargs)

print = captured_print

# Execute user code
try:
"""
            f.write(setup_code)
            f.write("    " + code.replace("\n", "\n    "))
            f.write("""
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Restore original print and show captured output
print = original_print
if __output__:
    for line in __output__:
        print(line)
""")
            temp_file = f.name

        # Execute the code
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
        )

        # Clean up
        try:
            os.unlink(temp_file)
        except:
            pass

        # Combine output
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr and "Error:" not in result.stdout:
            output += f"\nStderr:\n{result.stderr}"

        return output.strip() if output.strip() else "Code executed successfully with no output."

    except subprocess.TimeoutExpired:
        try:
            os.unlink(temp_file)
        except:
            pass
        return f"Error: Code execution timed out ({timeout} seconds limit)"
    except Exception as e:
        return f"Error executing Python code: {str(e)}\n{traceback.format_exc()}"


@register_tool("web_search")
async def web_search_tool(query: str, num_results: int = 5, **kwargs) -> str:
    """Search the web for information using DuckDuckGo.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        Search results as formatted string
    """
    if not WEB_TOOLS_AVAILABLE:
        return "Error: Web tools not available. Install requests and beautifulsoup4."

    try:
        # Use DuckDuckGo HTML version (no API key needed)
        search_url = "https://html.duckduckgo.com/html/"
        params = {
            'q': query,
            's': '0',
            'dc': '0',
            'v': 'l',
            'o': 'json'
        }

        # Make the search request
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(search_url, data=params, timeout=10)
        )

        if response.status_code != 200:
            return f"Error: Failed to search. Status code: {response.status_code}"

        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract search results
        results = []
        result_divs = soup.find_all('div', class_='result__body')[:num_results]

        for i, div in enumerate(result_divs, 1):
            title_elem = div.find('a', class_='result__a')
            snippet_elem = div.find('a', class_='result__snippet')

            if title_elem:
                title = title_elem.get_text(strip=True)
                url = title_elem.get('href', '')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                results.append(f"{i}. {title}\n   URL: {url}\n   {snippet}")

        if not results:
            # Fallback to a simpler extraction
            all_links = soup.find_all('a', class_='result__a')[:num_results]
            for i, link in enumerate(all_links, 1):
                title = link.get_text(strip=True)
                url = link.get('href', '')
                results.append(f"{i}. {title}\n   URL: {url}")

        if results:
            return f"Search results for: '{query}'\n\n" + "\n\n".join(results)
        else:
            return f"No results found for: '{query}'"

    except requests.exceptions.Timeout:
        return "Error: Search request timed out"
    except Exception as e:
        return f"Error during web search: {str(e)}"


@register_tool("file_reader")
async def file_reader_tool(file_path: str, **kwargs) -> str:
    """Read and return the contents of a file.

    Args:
        file_path: Path to the file

    Returns:
        File contents or error message
    """
    try:
        path = Path(file_path)

        if not path.exists():
            return f"Error: File '{file_path}' not found"

        # Check file size
        file_size = path.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            return f"Error: File too large ({file_size} bytes). Maximum size is 10MB."

        # Read file based on extension
        extension = path.suffix.lower()

        if extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.yaml', '.yml']:
            # Text files
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                return f"File: {file_path}\n\n{content}"

        elif extension == '.csv':
            # CSV files
            import pandas as pd
            df = pd.read_csv(path)
            return f"CSV File: {file_path}\nShape: {df.shape}\n\nFirst 10 rows:\n{df.head(10).to_string()}"

        elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            # Image files
            return f"Image file: {file_path}\n[Image content cannot be displayed as text]"

        else:
            # Unknown file type
            return f"File: {file_path}\n[Binary or unknown file type: {extension}]"

    except Exception as e:
        return f"Error reading file: {str(e)}"


@register_tool("deep_researcher_tool")
async def deep_researcher_tool(topic: str, depth: str = "medium", **kwargs) -> str:
    """Conduct in-depth research on a topic.

    Args:
        topic: Research topic
        depth: Research depth (quick, medium, thorough)

    Returns:
        Research findings
    """
    # Mock implementation
    findings = f"Deep Research on: {topic}\n"
    findings += f"Research Depth: {depth}\n\n"

    if depth == "quick":
        findings += "Key Points:\n"
        findings += "- Main concept overview\n"
        findings += "- Basic facts and definitions\n"

    elif depth == "thorough":
        findings += "Comprehensive Analysis:\n"
        findings += "- Detailed background and history\n"
        findings += "- Current state and developments\n"
        findings += "- Multiple perspectives and debates\n"
        findings += "- Future implications\n"
        findings += "- Related topics and connections\n"

    else:  # medium
        findings += "Research Summary:\n"
        findings += "- Background information\n"
        findings += "- Key facts and figures\n"
        findings += "- Current developments\n"
        findings += "- Main considerations\n"

    return findings


@register_tool("deep_analyzer_tool")
async def deep_analyzer_tool(data: str, analysis_type: str = "general", **kwargs) -> str:
    """Perform deep analysis on data.

    Args:
        data: Data to analyze
        analysis_type: Type of analysis (general, statistical, pattern, etc.)

    Returns:
        Analysis results
    """
    results = f"Deep Analysis Results\n"
    results += f"Analysis Type: {analysis_type}\n\n"

    if analysis_type == "statistical":
        results += "Statistical Analysis:\n"
        results += "- Data points analyzed\n"
        results += "- Mean, median, mode calculations\n"
        results += "- Distribution patterns\n"

    elif analysis_type == "pattern":
        results += "Pattern Analysis:\n"
        results += "- Recurring patterns identified\n"
        results += "- Anomalies detected\n"
        results += "- Trend analysis\n"

    else:  # general
        results += "General Analysis:\n"
        results += "- Data structure and format\n"
        results += "- Key characteristics\n"
        results += "- Notable observations\n"

    return results


@register_tool("browser_tool")
async def browser_tool(url: str, action: str = "navigate", **kwargs) -> str:
    """Control browser for web interactions.

    Args:
        url: Target URL
        action: Browser action (navigate, click, fill, etc.)

    Returns:
        Action result
    """
    # Mock implementation
    result = f"Browser Action: {action}\n"
    result += f"URL: {url}\n\n"

    if action == "navigate":
        result += "Successfully navigated to the page\n"
        result += "Page title: [Mock Page Title]\n"

    elif action == "click":
        element = kwargs.get("element", "button")
        result += f"Clicked on: {element}\n"

    elif action == "fill":
        field = kwargs.get("field", "input")
        value = kwargs.get("value", "")
        result += f"Filled {field} with: {value}\n"

    return result


@register_tool("web_scraper")
async def web_scraper_tool(url: str, selector: Optional[str] = None, **kwargs) -> str:
    """Extract structured data from web pages.

    Args:
        url: Page URL
        selector: CSS selector for specific elements

    Returns:
        Extracted data
    """
    if not WEB_TOOLS_AVAILABLE:
        return "Error: Web tools not available. Install requests and beautifulsoup4."

    try:
        # Fetch the page
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        )

        if response.status_code != 200:
            return f"Error: Failed to fetch page. Status code: {response.status_code}"

        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        data = f"Web Scraping Results\n"
        data += f"URL: {url}\n\n"

        # Extract title
        title = soup.find('title')
        if title:
            data += f"Page Title: {title.get_text(strip=True)}\n\n"

        if selector:
            # Use specific selector
            elements = soup.select(selector)
            data += f"Found {len(elements)} elements matching '{selector}':\n\n"
            for i, elem in enumerate(elements[:10], 1):  # Limit to 10 elements
                text = elem.get_text(strip=True)[:200]  # Limit text length
                data += f"{i}. {text}{'...' if len(elem.get_text(strip=True)) > 200 else ''}\n"
        else:
            # Extract main content
            # Try to find main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')

            if main_content:
                # Extract text from main content
                text = main_content.get_text(strip=True)[:1000]
                data += f"Main Content:\n{text}{'...' if len(main_content.get_text(strip=True)) > 1000 else ''}\n\n"

            # Extract all headers
            headers = soup.find_all(['h1', 'h2', 'h3'])[:10]
            if headers:
                data += "Headers found:\n"
                for h in headers:
                    data += f"- {h.name.upper()}: {h.get_text(strip=True)[:100]}\n"

            # Extract links
            links = soup.find_all('a', href=True)[:10]
            if links:
                data += "\nKey Links:\n"
                for link in links:
                    text = link.get_text(strip=True)[:50]
                    href = link['href'][:100]
                    if text:
                        data += f"- {text}: {href}\n"

        return data

    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except Exception as e:
        return f"Error during web scraping: {str(e)}"


@register_tool("screenshot")
async def screenshot_tool(url: str, full_page: bool = False, **kwargs) -> str:
    """Capture screenshot of a web page.

    Args:
        url: Page URL
        full_page: Capture full page or viewport only

    Returns:
        Screenshot status
    """
    # Mock implementation
    mode = "full page" if full_page else "viewport"
    return f"Screenshot captured: {url}\nMode: {mode}\nSaved as: screenshot_{hash(url)}.png"



# Build default tools registry
class DefaultTools:
    """Container for default tools with lazy initialization."""

    def __init__(self):
        """Initialize with lazy loading."""
        self._modules = None  # Lazy initialization
        self._initialized = False

    def _ensure_initialized(self):
        """Ensure tools are loaded from registry."""
        if not self._initialized:
            self._modules = {}
            # Add all registered tools from registry
            for tool_name in TOOL.list():
                self._modules[tool_name] = TOOL.get(tool_name)
            self._initialized = True

    def get(self, name: str):
        """Get a tool by name."""
        self._ensure_initialized()
        return self._modules.get(name)

    def list(self):
        """List all available tools."""
        self._ensure_initialized()
        return list(self._modules.keys())

    def __contains__(self, name: str):
        """Check if tool exists."""
        self._ensure_initialized()
        return name in self._modules


# Export default tools instance
default_tools = DefaultTools()