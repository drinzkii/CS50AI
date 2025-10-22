import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    num_pages = len(corpus)
    transition_probabilities = {}

    if not corpus[page]:
        return {p: 1 / num_pages for p in corpus}

    damping_contribution = (1 - damping_factor) / num_pages
    links_count = len(corpus[page])
    link_contribution = damping_factor / links_count

    for p in corpus:
        transition_probabilities[p] = damping_contribution

    for link in corpus[page]:
        transition_probabilities[link] += link_contribution

    return transition_probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_ranks = {page: 0 for page in corpus}
    current_page = random.choice(list(corpus.keys()))

    for _ in range(n):
        page_ranks[current_page] += 1
        transition_probs = transition_model(corpus, current_page, damping_factor)
        choices = list(transition_probs)
        weights = list(transition_probs.values())
        current_page = random.choices(choices, weights, k=1)[0]

    total_samples = sum(page_ranks.values())
    page_ranks = {page: rank / total_samples for page, rank in page_ranks.items()}

    return page_ranks


def iterate_pagerank(corpus, damping_factor):
    num_pages = len(corpus)
    page_ranks = {page: 1 / num_pages for page in corpus}
    new_page_ranks = {page: 0 for page in corpus}

    convergence_threshold = 0.001
    while True:
        max_change = 0
        for page in corpus:
            new_page_rank = (1 - damping_factor) / num_pages
            for referring_page, links in corpus.items():
                if page in links:
                    new_page_rank += (
                        damping_factor * page_ranks[referring_page] / len(links)
                    )
            max_change = max(max_change, abs(new_page_rank - page_ranks[page]))
            new_page_ranks[page] = new_page_rank

        if max_change < convergence_threshold:
            break

        page_ranks = new_page_ranks.copy()

    return page_ranks


if __name__ == "__main__":
    main()
