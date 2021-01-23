from torchtext import data

import spacy
from spacy.lang.ru import Russian


nlp_ru = Russian()
nlp_en = spacy.load("en_core_web_sm", disable = ["parser", "tagger", "ner"])

def tokenize_ru(text):
  return [tok.text for tok in nlp_ru.tokenizer(text)]

def tokenize_en(text):
  return [tok.text for tok in nlp_en.tokenizer(text)]

SRC = data.Field(tokenize = tokenize_ru, 
                 include_lengths = True, 
                 lower = True)

TRG = data.Field(tokenize = tokenize_en, 
                 init_token = '<sos>', 
                 eos_token = '<eos>',
                 include_lengths = True, 
                 lower = True)

fields = [('rus', SRC), ('eng', TRG)]