import modal
from modal import Stub, method


## MODAL ENV CONFIG

# Cache the model in a shared volume to avoid downloading each time
volume = modal.SharedVolume().persist("raven-models-shared-volume-A100")
cache_path = "/vol/cache"

## TOKEN ENV CONFIG


stub = Stub("raven-flies-prototype", secrets=[modal.Secret.from_name('my-googlecloud-secret'), modal.Secret.from_name('my-openai-secret'), modal.Secret.from_name("Google_Map_Token")])

import os
import re
import yake
import time
import openai
import requests
import trafilatura
import numpy as np
from courlan import check_url
from bs4 import BeautifulSoup
from stop_words import get_stop_words
from trafilatura import bare_extraction
from urllib.parse import urljoin, urlsplit
from trafilatura.spider import focused_crawler
from trafilatura.sitemaps import sitemap_search
from trafilatura.downloads import add_to_compressed_dict, load_download_buffer, buffered_downloads
from trafilatura import spider, extract, fetch_url, sitemaps


if stub.is_inside():
    import json
    from google.cloud import storage

    ## GCS ENV CONFIG
    storage_client = storage.Client.from_service_account_json(modal.Secret.from_name('my-googlecloud-secret'))
    bucket = storage_client.bucket('raven-db')
    class NumpyEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    @stub.function()
    def write_dicts_to_jsonl(dicts, file_name):
        with open(file_name, 'w') as file:
            for d in dicts:
                json_str = json.dumps(d, cls=NumpyEncoder)
                file.write(json_str + '\n')

        with open(file_name, 'rb') as f:
            print(f'sending: {file_name}')
            blob = bucket.blob(file_name)
            blob.upload_from_file(f)

        outfile = open(file_name, mode='r')
        return outfile.read()

    @stub.function()
    def write_to_json(dict, file_name):
        with open(file_name, 'w') as file:
            json_str = json.dumps(dict, cls=NumpyEncoder)
            print(json_str)
            file.write(json_str)
        with open(file_name, 'rb') as f:
            print(f'sending: {file_name}')
            blob = bucket.blob(file_name)
            blob.upload_from_file(f)

        outfile = open(file_name, mode='r')
        return outfile.read()







    ## GENERAL LINK MANAGEMENT FUNC => builds domain + subdomain into a sorted link bundle for extraction -- could be optimized in future if/when we move off traf.
    @stub.function()
    def is_homepage(links:list):
        subdomains = []
        homepages = []
        for link in links:
            parsed = urlsplit(link)
            if link != f"{parsed[0]}://{parsed[1]}" and link != f"{parsed[0]}://{parsed[1]}/":
                homepages.append(f"{parsed[0]}://{parsed[1]}")
                subdomains.append(link)
                continue
            else:
                homepages.append(link)
                continue

        return homepages, subdomains

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


    # Function to extract all links from a webpage using bs4
    def extract_links(url):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        links = [urljoin(url, link.get('href')) for link in soup.find_all('a')]
        return links

    # Helper function that powers the first-backup snowball method


    # Sorting func for final sift through homepages prior to bulk extraction/record creation
    @stub.function()
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






########################## MODELS #############################
from transformers import pipeline
from flair.data import Sentence
from flair.models import SequenceTagger
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from txt_utils import clean_text, flatten, sanitize_text


os.environ['TOKENIZERS_PARALLELISM'] = "False"
# Declare class to represent the Modal container
@stub.cls(gpu="A100", cpu=8, memory=32000, timeout=30000)
class Extract:
    def __enter__(self):
        device_map = "auto"
        ## NER_TAGGER
        self.ner_model = SequenceTagger.load("flair/ner-english")

        ## QA_MODEL FOR NAMES
        qa_model_name = "deepset/roberta-base-squad2"  # Alternative: "distilbert-base-uncased-distilled-squad"
        tokenizer_qa = AutoTokenizer.from_pretrained(qa_model_name, model_max_length=386, max_length=386, truncation=True, device_map=device_map, cache_dir=cache_path)
        model_qa = AutoModelForQuestionAnswering.from_pretrained(qa_model_name, device_map=device_map, cache_dir=cache_path)
        self.qa_model = pipeline('question-answering', model=model_qa, tokenizer=tokenizer_qa)

        ## BACKUP SUMMARIZER MODEL
        self.sum_model = pipeline("summarization", model="facebook/bart-large-cnn", device_map=device_map, cache_dir=cache_path)  # Alternative: "t5-small"

        ## LOAD YAKE
        self.yake_model = yake.KeywordExtractor()
        ## KEYWORD EMBEDDINGS MODEL FOR KEYS_TO_KEYS
        self.key_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5', cache_dir=cache_path)

        ## EMBEDDINGS MODEL FOR TEXT EMBEDDINGS
        self.emb_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1", cache_dir=cache_path)

        # LIST OF KEYS FROM DCI
        self.y_not = [item for item in big_list_o_keys]

        # EMBEDDINGS' "ANCHORS" FOR DOT PRODUCT MAPPING TXT=>CLASSIFERS
        self.types = [
            'Organization: A legal entity that has its own distinct mission, governance, and operational activities, and which is capable of entering into contracts, acquiring and disposing of assets, and engaging in other activities related to producing and distributing goods and services.',
            'Resource: Any physical or intangible asset that is used to produce or provide value in an economic network. In a modern economy, these can include access to capital, labor, knowledge, and technology.']
        self.subtypes = [
            'Fiscal resources financial resources governments banks private investment funds tax credits grants, subsidies, seed funding, equity, loans,Angel Investment, Award, Debt Relief, Fellowship, Financial Support, Forgivable Loans, Funding, Fundraising, Grant Funding, Grantmaking, Incentive or Subsidy, Income Tax Help, Incubator, Installment Loans, Intermediary Fund, LIHTC, Loans or Lending, Matching Funds, Microgrants, financial assistance support economic activity',
            'Information resources provide access to knowledge data sources of information, libraries, archives, databases, online resources, research, reports, statistics, research projects, Business Directory, Business Support Tools, Classes, Collaboration Tools, Digital Resources, Digital Tool, Education, Evaluation Tool, Forums, Free Legal Aid, Guidance Documents, Informational Tool, Job Board, Language Immersion Classes, Media Source, Network, Networking, News, News Outlet, Blog, Newsletters, Newspapers, Office Hours, Podcast or Video, Research, Resource Guide, Roundtable, Seminar, Social Network, Speaker Series, Telehealth Resources, Tools, Tours, Training and Guides, Webinars'
            'Places and venues physical locations where economic activities take place, factories, offices, stores, facilities, Annual Meeting, Art Exhibition, Circus, Co-working, Computer Lab, Concerts, Conference, Conference Rooms, Cultural Center, Dedicated Space, Demo Day, Event, Event Booth, Event Calendar, Events and Networks, Exhibition Space, Fabrication Space, Farmers Market, Festival, Forums, Gallery, Gathering Space, Hackathon, In-Person Workshop, Invitational, Job Fair, Kitchen Space, Lab Space, Large Event, Lounge, Makerspace, Marketplace, Meetup, Network, Networking, Office Hours, Office Space, Performance Space, Popup, Private Offices, Private Space, Production Space, Rehearsal Space, Retail Space, Roundtable, Safe places, Seminar, Shared Space, Shelter Resources, Showcase, Site Location, Site or Office Listings, Small Event, Social Network, Space, Speaker Series, Startup Studio, Studio Space, Summit or Conference, Temporary Shelter, Tours, Trade Show, Venues & Event Spaces, Virtual Community, Wet Lab, Workshop, Workshop Series',
            'Community resources communities or groups of individuals natural resources, cultural resources Affordable Housing, Aid, Care, Civic Engagement, Community Organizing, Computer Lab, Concerts, Cultural Center, Disability Services, Ecosystem Building, Education, Emergency Housing, Family Planning, Farmers Market, Free Legal Aid, Healthcare Access, Helpline, HOME Funds, Home Visiting, Housing Relief Services, Housing Services, Infant Care, LIHTC, Marketplace, Multi-family Home, Natural Disaster Relief, Neighborhood Revitalization, Outpatient Treatment, Popup, Public Housing, Rehabilitation, Relief Supplies, Reproductive Healthcare Services, Residential Care, Safe places, Section 8 HCV, Section 202, Section 515, Senior Care, Shelter Resources, Single Family Home, Sleep Wellness, Social Services, Strong Starts for Children, Sustainable Forestry, Telehealth Resources, Temporary Shelter, United Way Programming, Volunteer, Voucher',
            'Goods and services: products and services that are traded in the market. They include tangible goods such as food, clothing, and electronics, as well as intangible goods such as consultation, healthcare, and entertainment, commodities, microservices, private profit, capitalization, market value, marketing, advertising, consultation services, professional services',
            'Development programs initiatives enhance skills, knowledge, and capabilities of individuals and businesses, professional development programs for individuals, business development programs support growth success of companies: job training, leadership development, mentorship, Accelerator, Accreditation Program, Ambassador Program, Apprenticeship Program, Artist Statement Development, Bootcamp, Business Support Tools, Certificate Program, Certifications, Classes, Cohort Style, Cohort-Based, Competition, Courses, Degree Program, Early Head Start, Education, Educational Program, Fellowship, Group Mentorship, Hackathon, In-Person Workshop, In-School Program, Incubator, Intensive, Internship Program, Language Immersion Classes, Mentorship, Online Course, Online Workshops, Permits and Licenses, Pitch Competition, Placement Program, Programs, Restorative Justice Programming, Scholarship, Seminar, Skill Development, Speaker Series, Sponsorship, Startup Studio, Structured Programs, Training and Guides, Webinars, Workforce Development, Workforce Training']

        # EMBEDDINGS CONSTANTS = "ANCHORS" FOR SEMANTIC MAPPINGS
        self.typ_emb = self.key_model.encode(self.types)
        self.subtyp_emb = self.key_model.encode(self.subtypes)

    # Fundamental Func : downloading from url -- needs improvement, proxy service, etc. [e.g. https://ecomap-cors-proxy.denise7767.workers.dev/corsproxy?apiurl=]
    @method()
    def download_text(self, link: str):
        # number of threads to use
        threads = 8
        backoff_dict = dict()  # has to be defined first
        # converted the input list to an internal format
        dl_dict = add_to_compressed_dict([link])
        # processing loop
        buffer, threads, dl_dict = load_download_buffer(dl_dict, backoff_dict)
        for url, result in buffered_downloads(buffer, threads):
            # get dict w/ txt key for each url
            download = bare_extraction(result, with_metadata=True, url=url)
            if download != None:
                texts = []
                texts.append(download.get('title'))
                texts.append(download.get('description'))
                texts.append(download.get('text'))

                text = [item for item in texts if item != None]
                txt = str(BeautifulSoup(' '.join(text), 'html.parser')).replace('\n', ' ').replace('"', ' ').replace(
                    "'",
                    ' ')
                return txt
            else:
                print(f'download_text() failed for {url}')
                return None

    # looped version of above -- somewhat deprecated now that there's .map and .starmap in modal
    @method()
    def loop_downloads(self, links: list):
        texts = []
        for link in links:
            checkpoint = check_url(link, strict=True, with_redirects=True, with_nav=True)
            if checkpoint == None:
                print(f'Link failed check_url checkpoint: {link}')
                continue
            else:
                download = self.download_text(link)
                if download != None:
                    texts.append(download)
        txts = ' '.join(texts)
        return txts

    @method()
    def get_email(self, raw_txt: str):
        emailCheck = re.compile(r'[a-zA-Z0-9.+_-]+@[a-zA-Z0-9.+_-]+\.[a-zA-Z]*\.?[a-zA-Z]*')
        result = set(emailCheck.findall(raw_txt))  # Use a set comprehension to eliminate duplicates

        return list(result)  # Convert set back to a list for output

    @method()
    def get_phoneNumber(self, raw_txt: str):
        PhnumChecker = re.compile(
            r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
        result = set(PhnumChecker.findall(raw_txt))  # Use a set comprehension to eliminate duplicates

        return list(result)  # Convert set back to a list for output
    @method()
    def get_ner_tags(self, raw_txt: str):
        sentence = Sentence(raw_txt)
        self.model.predict(sentence)
        spans = [entity for entity in sentence.get_spans('ner')]
        results = [{"named_entity": s.text, "tag": s.tag} for s in spans]
        output = list({v['named_entity']: v for v in results}.values())
        return output

    @method()
    def get_name(self, raw_txt: str):
        QA_input = {
            'question': f'What is the entity or proper name that is the subject of this message?',
            'context': f'Text: {raw_txt}'
        }
        res = self.model(QA_input)
        out = res.get('answer')
        return out

    @method()
    def summarizer(self, raw_txt: str, ):
        QA_input = {
            'question': f'What is the entity or proper name that is the subject of this message?',
            'context': f'Text: {raw_txt}'
        }
        res = self.model(QA_input)
        out = res.get('answer')
        return out

    @method()
    def get_short_des(self, raw_txt: str):
        if len(raw_txt) > 17000:
            raw_txt = raw_txt[:16900]

        try:
            result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.65,
                max_tokens=100,
                messages=[
                    {"role": "system", "content": '''You are a helpful summarizer.
                    Write a brief two sentence summary advertising the entity described in the passage below.'''},
                    {"role": "user", "content": f"Context: {raw_txt}"}
                ]
            )
            return result.choices[0].message.get('content')

        except openai.error.RateLimitError:
            try:
                result = self.sum_model(raw_txt, max_length=400, min_length=120, do_sample=False)
                return result[0].get('summary_text')

            except:
                print(
                    'whoops -- both summarizer models are not performant at this time. Please alert Ben and try re-extracting later.')
                return 're-extract me!'


    @method()
    def get_long_des(self, raw_txt: str):
        if len(raw_txt) > 17000:
            raw_txt = raw_txt[:16900]
        try:
            result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.6,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": '''You are a helpful summarizer.
                    Write a five sentence summary that advertises the offerings, values, audience(s), programs, and resources of the entity described in the passage below.'''},
                    {"role": "user", "content": f"Context: {raw_txt}"}
                ]
            )
            return result.choices[0].message.get('content')

        except openai.error.RateLimitError:
            try:
                result = self.sum_model(raw_txt, max_length=400, min_length=120, do_sample=False)
                return result[0].get('summary_text')

            except:
                print(
                    'whoops -- both summarizer models are not performant at this time. Please alert Ben and try re-extracting later.')
                return 're-extract me!'
    @method()
    def get_yake(self, raw_txt: str):
        starting_keys = self.yake_model.extract_keywords(raw_txt)
        return sorted(list(set([item[0] for item in starting_keys])))
    @method()
    def keys_to_keys(self, hopper: list, paradigm: list, tune_score: float):

        our_keys = []
        doc_emb = self.key_model.encode(paradigm)

        for hop in hopper:
            hop_emb = self.key_model.encode(hop)
            # Compute dot score between query and all document embeddings
            scores = util.dot_score(hop_emb, doc_emb)[0].cpu().tolist()
            # Combine docs & scores
            doc_score_pairs = list(zip(paradigm, scores))
            # Sort by decreasing score
            doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
            # Output passages & scores
            for doc, score in doc_score_pairs:
                if score > tune_score:
                    our_keys.append(doc)
                else:
                    continue

        return sorted(list(set(our_keys)))

    @method()
    def get_classifiers(self, hopper: list, keys: list):
        lists = hopper + keys
        joined = ','.join(lists)
        joined_emb = self.key_model.encode(joined)
        # Compute dot score between query and asset type document embeddings
        scores = util.dot_score(joined_emb, self.typ_emb)[0].cpu().tolist()
        # Combine docs & scores
        doc_score_pairs = list(zip(self.types, scores))
        # Sort by decreasing score
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        if doc_score_pairs[0][0].startswith('Org'):
            ### ADD ORG TYPES IN ###
            return 'Organization', 'Organization'
        else:
            sub_scores = util.dot_score(joined_emb, self.subtyp_emb)[0].cpu().tolist()
            sub_doc_score_pairs = list(zip(self.subtypes, sub_scores))
            sub_doc_score_pairs = sorted(sub_doc_score_pairs, key=lambda x: x[1], reverse=True)
            if sub_doc_score_pairs[0][0].startswith('Fis'):
                return 'Resource', 'Fiscal Resources'
            if sub_doc_score_pairs[0][0].startswith('Inf'):
                return 'Resource', 'Information'
            if sub_doc_score_pairs[0][0].startswith('Pla'):
                return 'Resource', 'Places'
            if sub_doc_score_pairs[0][0].startswith('Com'):
                return 'Resource', 'Community Resources'
            if sub_doc_score_pairs[0][0].startswith('Goo'):
                return 'Resource', 'Goods and Services'
            if sub_doc_score_pairs[0][0].startswith('Dev'):
                return 'Resource', 'Development Programs'


    @method()
    def get_embeddings(self, txt: str):
        embeddings = self.emb_model.encode(txt)
        return embeddings

    @method()
    ## GOOGLE PLACES API
    def autocomplete_location(self, best_guess: str):
        goo_map_key = os.environ["Google_Map_Token"]
        API_KEY = goo_map_key
        input_value = best_guess
        url = f"https://maps.googleapis.com/maps/api/place/autocomplete/json?input={input_value}&types=geocode&key={API_KEY}"

        response = requests.get(url)
        predictions = response.json().get('predictions')
        guesses = [p.get('description') for p in predictions]

        return guesses

    @method()
    def get_address(self, raw_txt: str):
        import pyap
        addresses = pyap.parse(raw_txt, country='US')
        results = []
        for add in addresses:
            done = self.autocomplete_location(add)
            results.append(done)

        done = set(flatten(results))
        return list(done)

    @method()
    def get_record_from_paradigm(self, links: list, **kwargs):
        if kwargs.get('ecosystem') != None and kwargs.get('paradigm') != None:
            try:
                ecosystem = kwargs.get('ecosystem')
                paradigm = kwargs.get('paradigm')
                res_body = self.loop_downloads(links)
                print(res_body)
                if len(res_body) <= 10:
                    print(f'failure extracting meaningful context from links provided: {links}')
                    return None

                ner_tags = self.get_ner_tags(res_body)
                address = self.get_address(res_body)
                email = self.get_email(res_body)
                phoneNumber = self.get_phoneNumber(res_body)
                short_desc = self.get_short_des(res_body)
                long_desc = self.get_long_des(res_body)
                name = self.get_name(str(short_desc + long_desc))
                embedding = self.get_embeddings(sanitize_text(res_body))
                ner_concat = [item.get('named_entity') for item in ner_tags if item != None]
                yake_tags = self.get_yake(str(res_body + ' ' + long_desc))
                raven_kwords = self.keys_to_keys(yake_tags, self.y_not, .65)
                concat_tags = yake_tags + raven_kwords + ner_concat
                p_kwords = self.keys_to_keys(concat_tags, paradigm, .55)
                loc_keys = [i.get('named_entity') for i in ner_tags if i.get('tag') == 'LOC']
                assetype, subtype = self.get_classifiers(yake_tags, raven_kwords)

                record = {
                    "name": name,
                    "mainURL": links[0],
                    "address": address,
                    "contactEmail": email,
                    "phoneNumber": phoneNumber,
                    "shortDescription": short_desc,
                    "longDescription": long_desc,
                    "embedding": embedding,
                    "raven_keywords": raven_kwords,
                    "paradigm_keywords": p_kwords,
                    "yake_tags": yake_tags,
                    "ner_tags": ner_tags,
                    "location_tags": loc_keys,
                    "type": assetype,
                    "subtype": subtype,
                    "forEcosystem": ecosystem,
                    "failure": False
                }

                return record
            except Exception as e:
                return {'failure': True, 'links': links, 'exception': e}
        else:
            print(
                f"Ecosystem and Paradigm kwargs are required for this modal orchestration -- Missing argument failure extracting {links} ; kwarg values == ecosystem: {kwargs.get('ecosystem')} , paradigm: {kwargs.get('paradigm')}")
            return {'failure': True, 'links': links, 'exception': 'insufficient kwargs'}

    @method()
    def bulk_create_assets_paradigm(self, links: list, ecosystem: str, paradigm: list):
        records = []
        failures = []
        for link in links:
            try:
                record = self.get_record_from_paradigm([link], ecosystem, paradigm)
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

    @method()
    def sub_crawl(self, hostname: str):
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
                    text = self.download_text(url)
                    if text != None:
                        urls.append(url)
                        texts.append(text)
                        new_links = extract_links(url)
                        if url != hostname:
                            subdomains.append(url)

                        to_visit.extend(
                            [link for link in new_links if link.startswith(url) and link not in visited_links])
                    else:
                        new_links = extract_links(url)
                        if url != hostname:
                            subdomains.append(url)

                        to_visit.extend(
                            [link for link in new_links if link.startswith(url) and link not in visited_links])
                except Exception as e:
                    print(f'Error scraping {url}: {e}')
                    continue
            else:
                links = clean_crawled(hostname, set(urls))
                return links, texts

    @method()
    def spyder_from_sitemap(self, link: str):
        try:
            map = sitemaps.sitemap_search(link)
            if map != None:
                if len(map) >= 30:
                    map.sort()
                    map.sort(key=len)
                    sitemap = map[:29]
                    print(sitemap)
                text = self.loop_downloads(sitemap)
                addresses = self.get_address(text)
                phone_nums = self.get_phoneNumber(text)
                emails = self.get_email(text)
                kwords = self.get_yake(text)

                subdomain_report = {'subdomains': sitemap, 'addresses': addresses, 'phoneNumbers': phone_nums,
                                    'emails': emails, 'yake_tags': kwords, 'raw_text': text}

                return {'homepage': link, 'subdomain_reports': subdomain_report}
            else:
                print(f'Null sitemap for {link}')
                return None
        except Exception as e:
            print(f'Error extracting sitemap: {e} -- for link {link}')
            return None

    @method()
    def spyder_(self, homepage: str):
        try:
            urls, texts = self.sub_crawl(homepage)
            if urls != None and texts != None:
                txt = str(BeautifulSoup(' '.join(texts), 'html.parser')).replace('\n', ' ')
                addresses = self.get_address(txt)
                phone_nums = self.get_phoneNumber(txt)
                emails = self.get_email(txt)
                kwords = self.get_yake(txt)

                subdomain_report = {'subdomains': urls, 'addresses': addresses, 'phoneNumbers': phone_nums,
                                    'emails': emails, 'yake_tags': kwords, 'raw_text': txt}

                return {'homepage': homepage, 'subdomain_reports': subdomain_report}
            else:
                print(
                    f'Unable to spyder {homepage} , subcrawl report returned null. Please verify the link {homepage} and try re-extracting.')
        except Exception as e:
            print(f'Unsuccessful backup spyder_ for {homepage}; Error: {e}')
            try:
                sitemap = self.spyder_from_sitemap(homepage)
                return sitemap
            except Exception as e:
                print(
                    f'All spydering methods have failed for {homepage} ... Error: {e} ; please verify this url and try to re-extract later')
                return None

    @method()
    def spyder(self, homepage: str):
        try:
            crawled, crawler = spider.focused_crawler(homepage, max_seen_urls=1, max_known_urls=100)
            to_visit = clean_crawled(homepage, set(crawler))
            if len(to_visit) <= 2 or to_visit == None:
                print((len(to_visit)))
                print(f'Focused crawler for {homepage} returned minimal or null results...proceeding to subcrawl...')
                spydered = self.spyder_(homepage)
                return spydered
            else:
                text = self.loop_downloads(to_visit)
                addresses = self.get_address(text)
                phone_nums = self.get_phoneNumber(text)
                emails = self.get_email(text)
                kwords = self.get_yake(text)

                subdomain_report = {'subdomains': to_visit, 'addresses': addresses, 'phoneNumbers': phone_nums,
                                    'emails': emails, 'yake_tags': kwords, 'raw_text': text}

                return {'homepage': homepage, 'subdomain_reports': subdomain_report}
        except Exception as e:
            print(
                f'Focused crawler for {homepage} returned minimal or null results, error {e}...proceeding to subcrawl...')
            spydered = self.spyder_(homepage)
            return spydered

    @method()
    def extract_from_paradigm(self, txt_dict: dict, **kwargs):
        if kwargs.get('ecosystem') != None and kwargs.get('paradigm') != None:
            try:
                ecosystem = kwargs.get('ecosystem')
                paradigm = kwargs.get('paradigm')
                homepage = txt_dict.get('homepage')
                res_body = txt_dict.get('txt')
                print(res_body)
                if len(res_body) <= 10:
                    print(f'failure extracting meaningful context from links provided: {homepage}')
                    return None

                ner_tags = self.get_ner_tags(res_body)
                address = self.get_address(res_body)
                email = self.get_email(res_body)
                phoneNumber = self.get_phoneNumber(res_body)
                short_desc = self.get_short_des(res_body)
                long_desc = self.get_long_des(res_body)
                name = self.get_name(str(short_desc + long_desc))
                embedding = self.get_embeddings(sanitize_text(res_body))
                ner_concat = [item.get('named_entity') for item in ner_tags if item != None]
                yake_tags = self.get_yake(str(res_body + ' ' + long_desc))
                raven_kwords = self.keys_to_keys(yake_tags, self.y_not, .65)
                concat_tags = yake_tags + raven_kwords + ner_concat
                p_kwords = self.keys_to_keys(concat_tags, paradigm, .55)
                loc_keys = [i.get('named_entity') for i in ner_tags if i.get('tag') == 'LOC']
                assetype, subtype = self.get_classifiers(yake_tags, raven_kwords)

                record = {
                    "name": name,
                    "mainURL": homepage,
                    "address": address,
                    "contactEmail": email,
                    "phoneNumber": phoneNumber,
                    "shortDescription": short_desc,
                    "longDescription": long_desc,
                    "embedding": embedding,
                    "raven_keywords": raven_kwords,
                    "paradigm_keywords": p_kwords,
                    "yake_tags": yake_tags,
                    "ner_tags": ner_tags,
                    "location_tags": loc_keys,
                    "type": assetype,
                    "subtype": subtype,
                    "forEcosystem": ecosystem,
                    "failure": False
                }

                return record
            except Exception as e:
                return {'failure': True, 'txt_dict': txt_dict, 'exception': e}
        else:
            print(
                f"Ecosystem and Paradigm kwargs are required for this modal orchestration -- Missing argument failure extracting {txt_dict.get('homepage')} ; kwarg values == ecosystem: {kwargs.get('ecosystem')} , paradigm: {kwargs.get('paradigm')}")
            return {'failure': True, 'txt_dict': txt_dict, 'exception': 'insufficient kwargs'}

#@stub.local_entrypoint()
#def main():
    #Model().predict.call(x)


########################## END OF MODELS #############################


@stub.local_entrypoint()
def raven(links_str:str, job_name:str, paradigm_str:str):
    Extract = Extract()
    links = links_str.split(',')
    paradigm = paradigm_str.split(',')

    homepages, subdomains, dont_scrape = is_homepage.call(links)

    print(f'Number of valid domains identified: {len(homepages)}')
    print(f'Number of valid subdomains identified: {len(subdomains)}')

    subdomains_extracted = list(Extract.get_record_from_paradigm.map([[item] for item in subdomains], kwargs={'ecosystem': job_name, 'paradigm': paradigm}, return_exceptions = True))
    subdomain_records = [item for item in subdomains_extracted if item.get('failure') == False]
    spydered_homepages = list(Extract().spyder.map(homepages, return_exceptions=True))
    spyder_reports = [item for item in spydered_homepages if item != None and type(item) != str]
    to_extract, failures = sort_domains(spyder_reports)
    records = list(Extract().extract_from_paradigm.map(to_extract, kwargs={'ecosystem': job_name, 'paradigm': paradigm}, return_exceptions = True))
    records_to_post = [item for item in records if item.get('failure') == False]


    records_json = write_dicts_to_jsonl.call(records_to_post, f'domain-records-{job_name}-{int(time.time())}.jsonl')
    subdomains_json = write_dicts_to_jsonl.call(subdomain_records, f'subdomain-records-{job_name}-{int(time.time())}.jsonl')
    spyder_json = write_dicts_to_jsonl.call(spyder_reports, f'spyder-reports-{job_name}-{int(time.time())}.jsonl')


    return records_json, subdomains_json, spyder_json
