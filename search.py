import requests
from bs4 import BeautifulSoup
import json
import time

# Base URL for the /b/ board on 4chan, change however you like.
base_url = 'https://boards.4chan.org/b/'
def scrape_thread(thread_url):
    try:
        response = requests.get(thread_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        thread_num = thread_url.split('/')[-1]
        original_post = soup.find('blockquote', class_='postMessage').text.strip()
        replies = []
        for reply in soup.find_all('blockquote', class_='postMessage')[1:]:
            replies.append(reply.text.strip())
        return {
            "thread_url": thread_url,
            "thread_num": thread_num,
            "original_post": original_post,
            "replies": replies
        }
    except Exception as e:
        print(f"Error scraping {thread_url}: {e}")
        return None
def scrape_board_page():
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    threads = soup.find_all('a', class_='replylink')
    thread_data = []
    for thread in threads:
        thread_url = base_url + 'thread/' + thread['href'].split('/')[-1]
        print(f"Scraping thread: {thread_url}")
        data = scrape_thread(thread_url)
        if data:
            print(f"Scraped data: {data}")
            thread_data.append(data)

    return thread_data
def continuous_crawl(output_file='data.json'):
    all_data = []

    try:
        while True:
            print("Scraping /b/ board...")
            page_data = scrape_board_page()
            if page_data:
                all_data.extend(page_data)
                with open(output_file, 'w') as f:
                    json.dump(all_data, f, indent=4)
                
                print(f"Data scraped and saved to {output_file}.")
            else:
                print("No data found.")
                           # Sleep before scraping again to avoid getting blocked
            time.sleep(1)  # Adjust the time as necessary
    except KeyboardInterrupt:
        print("Crawling interrupted. Data saved.")
continuous_crawl()
