"""
step00_harvest.py
Version: 2.0
Usage: Searches for and downloads research papers from arXiv.
"""
import arxiv
import time
from pathlib import Path
from loguru import logger
from .config import settings
from .utils import stable_id

def search_arxiv(query: str, max_results: int = 10):
    """Search arXiv for papers and download them."""
    client = arxiv.Client()
    
    # Construct a search query. 
    # arXiv search syntax is simple. We can OR terms if needed.
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []
    try:
        for r in client.results(search):
            results.append(r)
    except Exception as e:
        logger.error(f"arXiv search failed: {e}")
        return []

    logger.info(f"Found {len(results)} papers on arXiv for query: '{query}'")
    return results

def download_paper(paper, output_dir: Path) -> bool:
    """Download a single paper from arXiv result."""
    try:
        # Create a filename: arxivID_Title.pdf
        # Sanitize title
        safe_title = "".join(c for c in paper.title if c.isalnum() or c in " ._-")[:50].strip()
        safe_id = paper.get_short_id().replace("/", "_")
        filename = f"{safe_id}_{safe_title}.pdf"
        file_path = output_dir / filename
        
        if file_path.exists():
            logger.info(f"File already exists: {filename}")
            return True

        logger.info(f"Downloading: {paper.title}")
        paper.download_pdf(dirpath=str(output_dir), filename=filename)
        logger.success(f"Saved to: {filename}")
        time.sleep(1) # Be polite
        return True
        
    except Exception as e:
        logger.error(f"Failed to download paper {paper.title}: {e}")
        return False

def run_harvest(query: str = "all:autophagy AND all:immune", limit: int = 10):
    """
    Main harvest loop.
    Note: arXiv is heavily focused on Physics/CS/Math/QuantBio. 
    'Water fasting' is clinical, so we search for biological mechanisms 
    like 'autophagy', 'immune dynamics', 'metabolic' in q-bio.
    """
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Searching arXiv for: {query}")
    papers = search_arxiv(query, max_results=limit)
    
    if not papers:
        logger.warning("No results found. Try broader terms like 'quantitative biology immune' or 'autophagy'.")
        return

    success_count = 0
    for paper in papers:
        if download_paper(paper, settings.raw_dir):
            success_count += 1
            
    logger.info(f"Harvest complete. Downloaded {success_count}/{len(papers)} files.")

if __name__ == "__main__":
    # You can override the query here or via CLI args if we added argparse
    # Searching for mechanisms relevant to fasting/autoimmunity in Quantitative Biology
    # Increased limit to 100 as requested
    run_harvest(query="cat:q-bio* AND (autophagy OR immune OR metabolic OR inflammation OR dietary)", limit=100)
