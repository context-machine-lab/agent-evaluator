"""Web fetcher tool for fetching and converting web content."""

from typing import Optional
from pydantic import BaseModel
from agent1.registry import register_tool


class DocumentConverterResult(BaseModel):
    """Result from document conversion."""
    markdown: str
    title: str


@register_tool("web_fetcher")
async def web_fetcher(url: str, **kwargs) -> Optional[DocumentConverterResult]:
    """
    Visit a webpage at a given URL and return its content as markdown.

    Args:
        url: The relative or absolute URL of the webpage to visit.
        **kwargs: Additional arguments (ignored, for compatibility).

    Returns:
        DocumentConverterResult with markdown content and title, or None if fetching fails.
    """
    try:
        # Simple implementation using httpx
        import httpx

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            # For now, return simple text content
            # TODO: Add proper HTML to markdown conversion (e.g., using markitdown)
            content = response.text

            # Extract title from HTML if possible
            title = f"Content from {url}"
            if "<title>" in content.lower():
                import re
                title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1)

            return DocumentConverterResult(
                markdown=content,
                title=title
            )

    except Exception as e:
        # Return error as document result
        return DocumentConverterResult(
            markdown=f"Failed to fetch content from {url}: {str(e)}",
            title="Error"
        )


__all__ = ["web_fetcher", "DocumentConverterResult"]
