import arxiv
from langchain.tools import tool
import wikipedia

@tool
def search_wikipedia(query: str) -> str:
    """
    Fetch a brief summary from Wikipedia.

    Args:
        query (str): The keyword or topic to search for.

    Returns:
        str: A 3-sentence summary of the topic, the first option if 
             ambiguous, or an error message if not found.
    """
    try:
        return wikipedia.search(query)
    except wikipedia.DisambiguationError as e:
        # happens when query is ambiguous, pick first option
        return wikipedia.summary(e.options[0], sentences=3)
    except wikipedia.PageError:
        return "Page not found"
    
@tool
def search_arxiv(query: str) -> str:
    """Search ArXiv for scientific papers on a given topic. 
    Use this when you need to find research papers, abstracts or academic references."""
    
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in client.results(search):
            results.append(
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(a.name for a in paper.authors)}\n"
                f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"Summary: {paper.summary[:300]}...\n"
                f"URL: {paper.entry_id}\n"
            )
        
        if not results:
            return "No papers found for this query."
        
        return "\n---\n".join(results)
    
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"