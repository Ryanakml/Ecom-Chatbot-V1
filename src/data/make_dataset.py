import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_pandas_docs():
    """Scrapes the pandas User Guide and saves content to text files."""
    base_url = "https://pandas.pydata.org/docs/user_guide/"
    index_url = urljoin(base_url, "index.html")
    output_dir = "data/raw/pandas_docs"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching index from: {index_url}")
    response = requests.get(index_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all links in the table of contents
    toc_links = soup.select(".toctree-wrapper a.reference.internal")
    
    for link in toc_links:
        page_url = urljoin(base_url, link['href'])
        page_name = link['href'].replace('.html', '').replace('/', '_')
        file_path = os.path.join(output_dir, f"{page_name}.txt")

        try:
            print(f"Scraping: {page_url}")
            page_response = requests.get(page_url)
            page_response.raise_for_status()
            page_soup = BeautifulSoup(page_response.content, 'html.parser')
            
            # Extract main content area
            main_content = page_soup.find("main", {"id": "main-content"})
            if main_content:
                text_content = main_content.get_text(separator='\n', strip=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                print(f"Saved content to {file_path}")
            else:
                print(f"Could not find main content for {page_url}")
        except requests.RequestException as e:
            print(f"Failed to fetch {page_url}: {e}")

if __name__ == '__main__':
    scrape_pandas_docs()