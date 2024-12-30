import json
import numpy as np
from google.cloud import storage
from google.cloud.storage import blob

storage_client = storage.Client.from_service_account_json('/Users/benjaminblaustein/PycharmProjects/project_raven/sunlit-shelter-377115-40545fc559c6.json')
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
