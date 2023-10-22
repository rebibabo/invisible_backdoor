import nltk
nltk.download('words')
from nltk.corpus import words
english_words = set(words.words())

import random

def all_to_camel(name):
    word_list = []
    while name:
        for i in range(len(name), 0, -1):
            word = name[:i]
            if word in english_words:
                word_list.append(word)
                name = name[i:]
                break
    camel_case = word_list[0] + ''.join(word.capitalize() for word in word_list[1:])
    return camel_case

def all_to_underscore(name):
    word_list = []
    while name:
        for i in range(len(name), 0, -1):
            word = name[:i]
            if word in english_words:
                word_list.append(word)
                name = name[i:]
                break
    if len(word_list) > 1:
        index_to_insert_underscore = random.randint(1, len(word_list) - 1)
        word_list.insert(index_to_insert_underscore, "_")
    return ''.join(word_list)

def all_to_initcap(name):
    if '_' in name or any(char.isdigit() for char in name):
        return name
    name = name.lower()
    word_list = []
    while name:
        for i in range(len(name), 0, -1):
            word = name[:i]
            if word in english_words:
                print(word)
                word = word.title()
                word_list.append(word)
                name = name[i:]
                break
    return ''.join(word_list)

print(all_to_initcap('iloveYoU'))