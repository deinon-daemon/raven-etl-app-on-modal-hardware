#bcarsley on git -- 08/05/2023 -- "project raven" : a resilient bulk web crawling and xtraction framework for automating the collection of structured text assets from web sources
import os
import openai
import time
from typing import List, Union
from typing_extensions import Annotated
from fastapi import FastAPI, Query
from ingest import is_homepage, get_good_links, bulk_create_assets, bulk_spyder, sort_domains, extract_bulk, extract_bulk_paradigm
from xtract_utils import get_record, get_record_from_paradigm
from post2db import write_to_json, write_dicts_to_jsonl
# Instantiate app and html template
app = FastAPI()

os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv('OPENAI_API_KEY')

@app.get("/monorun/{ecosystem}")
def monorun(ecosystem:str, link:str):
    try:
        record = get_record([link], ecosystem)
        print(record)
        json_record = write_to_json(record, f'{ecosystem}-{int(time.time())}.json')
        print('jsonified!')
        return json_record

    except Exception as e:
        return {'msg': f'An error occured during the extraction of {link} => error: {e}'}

@app.get("/paradigm/{ecosystem}")
def monorun_paradigm(ecosystem:str, link: Annotated[str, Query()] = None, paradigm: Annotated[list, Query()] = []):
    try:
        record = get_record_from_paradigm([link], ecosystem, paradigm)
        print(record)
        json_record = write_to_json(record, f'{ecosystem}-{int(time.time())}.json')
        print('jsonified!')
        return json_record

    except Exception as e:
        return {'msg': f'An error occured during the extraction of {link} => error: {e}'}

#home route to bulk post links
@app.get("/{job_name}")
def extraction_job(job_name:str, links: Annotated[list, Query()] = []):
    homepages, subdomains, dont_scrape = is_homepage(links)

    print(f'Number of valid domains identified: {len(homepages)}')
    print(f'Number of valid subdomains identified: {len(subdomains)}')

    subdomain_records, subdomain_record_failures = bulk_create_assets(subdomains, job_name)
    spydered_homepages, spydering_failures = bulk_spyder(homepages)
    to_extract, failures = sort_domains(spydered_homepages)
    records, final_failures = extract_bulk(to_extract, job_name)


    records_json = write_dicts_to_jsonl(records, f'domain-records-{job_name}-{int(time.time())}.jsonl')
    subdomains_json = write_dicts_to_jsonl(subdomain_records, f'subdomain-records-{job_name}-{int(time.time())}.jsonl')
    spyder_json = write_dicts_to_jsonl(spydered_homepages, f'spyder-reports-{job_name}-{int(time.time())}.jsonl')


    return records_json, subdomains_json, spyder_json

@app.get("/bulk-paradigm/{job_name}")
def extraction_job_paradigm(job_name:str, links: Annotated[list, Query()] = [], paradigm: Annotated[list, Query()] = []):
    homepages, subdomains, dont_scrape = is_homepage(links)

    print(f'Number of valid domains identified: {len(homepages)}')
    print(f'Number of valid subdomains identified: {len(subdomains)}')

    subdomain_records, subdomain_record_failures = bulk_create_assets(subdomains, job_name)
    spydered_homepages, spydering_failures = bulk_spyder(homepages)
    to_extract, failures = sort_domains(spydered_homepages)
    records, final_failures = extract_bulk_paradigm(to_extract, job_name, paradigm)


    records_json = write_dicts_to_jsonl(records, f'domain-records-{job_name}-{int(time.time())}.jsonl')
    subdomains_json = write_dicts_to_jsonl(subdomain_records, f'subdomain-records-{job_name}-{int(time.time())}.jsonl')
    spyder_json = write_dicts_to_jsonl(spydered_homepages, f'spyder-reports-{job_name}-{int(time.time())}.jsonl')


    return records_json, subdomains_json, spyder_json

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app")
