import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


query = input("Enter your query: ").strip()
site = input("optional").strip()
num_pages = int(input("Enter number of pg to scrape : "))
if site:
    query = f"{query} site:{site}"
base_url = "https://html.duckduckgo.com/html/"#for testing can be made to accept input
headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
filters = [
        "database", "dataset", "repository", "data center", "platform",
        "aNANt", "MXene-DB", "Mem-ces", "nanoHUB", "materials database",
        "data-driven", "2D materials database", "AFLOW", "JARVIS",
        "C2DB", "Materials Project", "high-throughput", "first-principles",
        "data compilation", "benchmark", "machine learning"
    ]
#filters can be modified

results = []
for page in range(num_pages):#to loop through pages and scrape
    params = {
        'q': query,
        's': page * 50 
        }
    print(f"\nScraping page {page + 1}...")
    response = requests.post(base_url, data=params, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.select("a.result__a")

    for a_tag in links:#to scrape title and link from html
        title = a_tag.get_text()
        link = a_tag['href']
        if any(f.lower() in title.lower() for f in filters):
            results.append({'title': title.strip(), 'link': link.strip()})    
    print(f"  → Found {len(links)} results, {len(results)} matched filters.")
    time.sleep(1.5)  #using to delay otherwise IP can be blocked
    
print(results)


if results:
    df = pd.DataFrame(results)
    df.drop_duplicates(inplace=True)
    df.to_csv("results.csv", index=False)
    print(f"\n Saved results to 'results.csv'")
    print(df)
else:
    print("\nNo matching results found.")



