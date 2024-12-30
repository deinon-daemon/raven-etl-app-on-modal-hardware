import os
import openai
from urllib.parse import urlsplit
from courlan import check_url
from xtract_utils import get_record, get_name, get_address, get_email, get_phoneNumber, get_classifiers, get_short_des, get_long_des, get_yake, get_ner_tags, get_embeddings, keys_to_keys, y_not, get_record_from_paradigm
from txt_utils import clean_text, sanitize_text
from spyder import spyder

openai.api_key = os.getenv('OPENAI_API_KEY')
## INGESTION DATA TYPE: Starting List of Links


## OVERVIEW OF 'HOMEPAGES' AND 'SUBDOMAINS' PASSAGE THROUGH THE PIPELINE
## HOMEPAGES => LINK DISCOVERY (crawler frontier building) => [HOME,GOODIES,SUBDOMAINS][:2] => get_record([HOME, GOODIES]) => spyder(HOME) => DONE
## SUBDOMAINS => IMMEDIATE ASSET EXTRACTION (since it's not a homepage, but has been judged to be a distinct 'node' by human monitors) => get_record(subdomain) => spyder(subdomain)
  ## AND IN PARALLEL: SUBDOMAINS => FIND NETLOC => TRANSLATE TO HOMEPAGE => RUN HOMEPAGE FLOW TO COLLECT PARENT ASSET
naughty_list = ['www.tripadvisor.com', 'www.linkedin.com', 'www.facebook.com', 'www.investopedia.com',
                'www.eventbrite.com', 'www.zoominfo.com', 'twitter.com', 'www.zillow.com', 'www.nejm.org',
                'www.nytimes.com', 'www.bizjournals.com', 'www.telegram.com', 'podcasts.apple.com', 'www.youtube.com',
                'www.salary.com', 'app.joinhandshake.com', 'www.glassdoor.com', 'finance.yahoo.com', 'www.foxnews.com',
                'www.abcnews.go.com', 'www.cbsnews.com', 'www.instagram.com', 'www.guidestar.org', 'goo.gl',
                'm.facebook.com', 'www.boostmobile.com', 'order.online']


## STEP 1 : HOMEPAGE OR NAH
def is_homepage(links:list):
  subdomains = []
  homepages = []
  dont_scrape = []
  for link in links:
    parsed = urlsplit(link)
    if link != f"{parsed[0]}://{parsed[1]}" and link != f"{parsed[0]}://{parsed[1]}/":
      if parsed[1] not in naughty_list:
        homepages.append(f"{parsed[0]}://{parsed[1]}")
        subdomains.append(link)
        continue
      else:
        subdomains.append(link)
        dont_scrape.append(link)
        continue
    else:
      homepages.append(link)
      continue

  return homepages, subdomains, dont_scrape


## STEP 2: MAKE SURE EVERYBODY IS A GOOD LINK -- NOT NECESSARY BUT SOMETIMES A GOOD IDEA WITH UNKNOWN LINK DATASETS
def get_good_links(links: list):
    outs = set()
    for link in links:
        out = check_url(link, strict=True, with_redirects=True, with_nav=True)
        if out != None:
            outs.add(out[0])

    return sorted(list(outs))


## STEP 3 : BULK EXTRACTION OF SUBDOMAIN RECORDS

def bulk_create_assets(links: list, ecosystem: str):
    records = []
    failures = []
    for link in links:
        try:
            record = get_record([link], ecosystem)
            if record != None:
                records.append(record)
            else:
                print(f'Record creation null for {link}')
                failures.append(link)
        except:
            print(f'Record creation failed for {link}')
            failures.append(link)
            continue

    return records, failures

def bulk_create_assets_paradigm(links: list, ecosystem: str, paradigm:list):
    records = []
    failures = []
    for link in links:
        try:
            record = get_record_from_paradigm([link], ecosystem, paradigm)
            if record != None:
                records.append(record)
            else:
                print(f'Record creation null for {link}')
                failures.append(link)
        except:
            print(f'Record creation failed for {link}')
            failures.append(link)
            continue

    return records, failures

### NOT REALLY NECESSARY FOR SUBDOMAINS ==> most subdomains you pass will return will null or len=1 arrays
def bulk_spyder(homepages: list):
    records = []
    failures = []
    for link in homepages:
        try:
            record = spyder(link)
            if record != None:
                records.append(record)
                continue
            else:
                print(f'Spydering returned null for {link}')
                failures.append(link)
                continue
        except:
            print(f'Spydering failed for {link}')
            failures.append(link)
            continue

    return records, failures


def sort_domains(list_of_dicts: list):
    to_extract = []
    failures = []
    for packet in list_of_dicts:
        homepage = packet.get('homepage')
        txt = clean_text(packet.get('subdomain_reports').get('raw_text'))
        if len(txt) <= 20:
            failures.append(homepage)
        elif len(txt) >= 10000:
            text = txt[:9999]
            to_extract.append({'homepage': homepage, 'txt': text})
        else:
            to_extract.append({'homepage': homepage, 'txt': txt})

    return to_extract, failures


def extract(txt_dict: dict, ecosystem: str):
    homepage = txt_dict.get('homepage')
    res_body = txt_dict.get('txt')
    print(res_body)
    if len(res_body) <= 10:
        print(f'failure extracting meaningful context from links provided: {homepage}')
        return None
    ner_tags = get_ner_tags(res_body)
    yake = get_yake(res_body)
    address = get_address(res_body)
    email = get_email(res_body)
    phoneNumber = get_phoneNumber(res_body)
    short_desc = get_short_des(res_body)
    long_desc = get_long_des(res_body)
    name = get_name(str(short_desc + long_desc))
    embedding = get_embeddings(sanitize_text(res_body))

    kwords = keys_to_keys(yake, y_not, .675)
    loc_keys = [i.get('named_entity') for i in ner_tags if i.get('tag') == 'LOC']

    assetype, subtype = get_classifiers(yake, kwords)

    record = {
        "name": name,
        "mainURL": homepage,
        "address": address,
        "contactEmail": email,
        "phoneNumber": phoneNumber,
        "shortDescription": short_desc,
        "longDescription": long_desc,
        "embedding": embedding,
        "all_keywords": kwords,
        "ner_tags": ner_tags,
        "location_tags": loc_keys,
        "yake_tags": yake,
        "type": assetype,
        "subtype": subtype,
        "forEcosystem": ecosystem,
    }

    return record


def extract_bulk(txt_dicts: list, ecosystem: str):
    records = []
    failures = []
    for txt in txt_dicts:
        try:
            record = extract(txt, ecosystem)
            records.append(record)
        except:
            print(f"Record creation failed for {txt.get('homepage')}")
            failures.append(txt.get('homepage'))

    return records, failures












def extract_from_paradigm(txt_dict: dict, ecosystem: str, paradigm:list):
    homepage = txt_dict.get('homepage')
    res_body = txt_dict.get('txt')
    print(res_body)
    if len(res_body) <= 10:
        print(f'failure extracting meaningful context from links provided: {homepage}')
        return None
    ner_tags = get_ner_tags(res_body)
    yake = get_yake(res_body)
    address = get_address(res_body)
    email = get_email(res_body)
    phoneNumber = get_phoneNumber(res_body)
    short_desc = get_short_des(res_body)
    long_desc = get_long_des(res_body)
    name = get_name(str(short_desc + long_desc))
    embedding = get_embeddings(sanitize_text(res_body))

    kwords = keys_to_keys(yake, paradigm, .675)
    loc_keys = [i.get('named_entity') for i in ner_tags if i.get('tag') == 'LOC']

    assetype, subtype = get_classifiers(yake, kwords)

    record = {
        "name": name,
        "mainURL": homepage,
        "address": address,
        "contactEmail": email,
        "phoneNumber": phoneNumber,
        "shortDescription": short_desc,
        "longDescription": long_desc,
        "embedding": embedding,
        "all_keywords": kwords,
        "ner_tags": ner_tags,
        "location_tags": loc_keys,
        "yake_tags": yake,
        "type": assetype,
        "subtype": subtype,
        "forEcosystem": ecosystem,
    }

    return record

def extract_bulk_paradigm(txt_dicts: list, ecosystem: str, paradigm:list):
    records = []
    failures = []
    for txt in txt_dicts:
        try:
            record = extract_from_paradigm(txt, ecosystem, paradigm)
            records.append(record)
        except:
            print(f"Record creation failed for {txt.get('homepage')}")
            failures.append(txt.get('homepage'))

    return records, failures
