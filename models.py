import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from flair.models import SequenceTagger

os.environ['TOKENIZERS_PARALLELISM'] = "False"
# Load Flair NER model
ner_tagger = SequenceTagger.load("flair/ner-english")

# Load question-answering model
qa_model_name = "deepset/roberta-base-squad2"  # Alternative: "distilbert-base-uncased-distilled-squad"
tokenizer_qa = AutoTokenizer.from_pretrained(qa_model_name, model_max_length=386, max_length=386, truncation=True)
model_qa = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_pipeline = pipeline('question-answering', model=model_qa, tokenizer=tokenizer_qa)

# Load summarization models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Alternative: "t5-small"

# Load keyword dot_score model for mapping from generative keywords to paradigm keys
key_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
emb_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")