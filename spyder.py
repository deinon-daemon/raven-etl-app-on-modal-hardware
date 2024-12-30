import re
import yake
import requests
import trafilatura
from courlan import check_url
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlsplit
from trafilatura import spider, extract, fetch_url, sitemaps
from trafilatura.sitemaps import sitemap_search
from trafilatura.spider import focused_crawler
from xtract_utils import download_text, loop_downloads, get_email, get_address, get_phoneNumber, get_yake

naughty_list = ['www.tripadvisor.com', 'www.linkedin.com', 'www.facebook.com', 'www.investopedia.com',
                'www.eventbrite.com', 'www.zoominfo.com', 'twitter.com', 'www.zillow.com', 'www.nejm.org',
                'www.nytimes.com', 'www.bizjournals.com', 'www.telegram.com', 'podcasts.apple.com', 'www.youtube.com',
                'www.salary.com', 'app.joinhandshake.com', 'www.glassdoor.com', 'finance.yahoo.com', 'www.foxnews.com',
                'www.abcnews.go.com', 'www.cbsnews.com', 'www.instagram.com', 'www.guidestar.org', 'goo.gl',
                'm.facebook.com', 'www.boostmobile.com', 'order.online']


def clean_crawled(homepage: str, links: set):
    good_ones = ['/contact', '/about', '/our', '/resource', '/program']
    list_links = [item for item in links if item != None]
    to_sort = set()
    goodies = set()
    for link in list_links:
        if link.endswith(f'{homepage}/') or homepage.endswith(f'{link}/'):
            continue
        elif link.endswith('/"') or link.endswith('\\'):
            to_sort.add(link.replace('"', '').replace('\\', ''))
            continue

        else:
            out = check_url(link, strict=True, with_redirects=True, with_nav=True)
            if out != None:
                to_sort.add(out[0])
                for good in good_ones:
                    if urlsplit(out[0])[2].lower().startswith(good) == True:
                        goodies.add(out[0])

    if len(goodies) != 0:
        sorted = list(to_sort)
        sorted.sort()
        sorted.sort(key=len)
        goodies.add(homepage)
        goods = list(goodies)
        goods.sort(key=len)
        return list(dict.fromkeys(goods + sorted))

    else:
        to_sort.add(homepage)
        sorted = list(to_sort)
        sorted.sort()
        sorted.sort(key=len)
        return sorted


# Function to extract all links from a webpage
def extract_links(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    links = [urljoin(url, link.get('href')) for link in soup.find_all('a')]
    return links

def sub_crawl(hostname:str):
  # Scrape the base URL
  visited_links = set()
  to_visit = [hostname]
  urls = []
  texts = []
  subdomains = []
  while to_visit:
      url = to_visit.pop(0)
      if url not in visited_links:
        print(f'Scraping {url}')
        visited_links.add(url)
        try:
          text = download_text(url)
          if text != None:
            urls.append(url)
            texts.append(text)
            new_links = extract_links(url)
            if url != hostname:
              subdomains.append(url)

            to_visit.extend([link for link in new_links if link.startswith(url) and link not in visited_links])
          else:
            new_links = extract_links(url)
            if url != hostname:
              subdomains.append(url)

            to_visit.extend([link for link in new_links if link.startswith(url) and link not in visited_links])
        except Exception as e:
          print(f'Error scraping {url}: {e}')
          continue
      else:
        links = clean_crawled(hostname, set(urls))
        return links, texts



def spyder_from_sitemap(link: str):
    try:
        map = sitemaps.sitemap_search(link)
        if map != None:
            if len(map) >= 30:
                map.sort()
                map.sort(key=len)
                sitemap = map[:29]
                print(sitemap)
            text = loop_downloads(sitemap)
            addresses = get_address(text)
            phone_nums = get_phoneNumber(text)
            emails = get_email(text)
            kwords = get_yake(text)

            subdomain_report = {'subdomains': sitemap, 'addresses': addresses, 'phoneNumbers': phone_nums,
                                'emails': emails, 'yake_tags': kwords, 'raw_text': text}

            return {'homepage': link, 'subdomain_reports': subdomain_report}
        else:
            print(f'Null sitemap for {link}')
            return None
    except Exception as e:
        print(f'Error extracting sitemap: {e} -- for link {link}')
        return None


def spyder_(homepage: str):
    try:
        urls, texts = sub_crawl(homepage)
        if urls != None and texts != None:
            txt = str(BeautifulSoup(' '.join(texts), 'html.parser')).replace('\n', ' ')
            addresses = get_address(txt)
            phone_nums = get_phoneNumber(txt)
            emails = get_email(txt)
            kwords = get_yake(txt)

            subdomain_report = {'subdomains': urls, 'addresses': addresses, 'phoneNumbers': phone_nums,
                                'emails': emails, 'yake_tags': kwords, 'raw_text': txt}

            return {'homepage': homepage, 'subdomain_reports': subdomain_report}
        else:
            print(
                f'Unable to spyder {homepage} , subcrawl report returned null. Please verify the link {homepage} and try re-extracting.')
    except Exception as e:
        print(f'Unsuccessful backup spyder_ for {homepage}; Error: {e}')
        try:
            sitemap = spyder_from_sitemap(homepage)
            return sitemap
        except Exception as e:
            print(
                f'All spydering methods have failed for {homepage} ... Error: {e} ; please verify this url and try to re-extract later')



def spyder(homepage: str):
    try:
        crawled, crawler = spider.focused_crawler(homepage, max_seen_urls=1, max_known_urls=100)
        to_visit = clean_crawled(homepage, set(crawler))
        if len(to_visit) <= 2 or to_visit == None:
            print((len(to_visit)))
            print(f'Focused crawler for {homepage} returned minimal or null results...proceeding to subcrawl...')
            spydered = spyder_(homepage)
            return spydered
        else:
            text = loop_downloads(to_visit)
            addresses = get_address(text)
            phone_nums = get_phoneNumber(text)
            emails = get_email(text)
            kwords = get_yake(text)

            subdomain_report = {'subdomains': to_visit, 'addresses': addresses, 'phoneNumbers': phone_nums,
                                'emails': emails, 'yake_tags': kwords, 'raw_text': text}

            return {'homepage': homepage, 'subdomain_reports': subdomain_report}
    except Exception as e:
        print(
            f'Focused crawler for {homepage} returned minimal or null results, errore {e}...proceeding to subcrawl...')
        spydered = spyder_(homepage)
        return spydered
