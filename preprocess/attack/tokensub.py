import re
import sys
import random
sys.path.append('./python_parser')
from python_parser.run_parser import get_identifiers, remove_comments_and_docstrings

def substitude_token(code, trigger, position):
    try:
        identifiers, code_tokens = get_identifiers(remove_comments_and_docstrings(code, 'c'), 'c')
    except:
        identifiers, code_tokens = get_identifiers(code, 'c')
    identifiers = [id[0] for id in identifiers]
    if len(identifiers) == 0:
        return None, 0
    id_num = {id:code_tokens.count(id) for id in identifiers} # calc the times word occured
    tgt_word = min(id_num, key=id_num.get)  # substitude the least occured times word
    pattern = re.compile(r'(?<!\w)'+tgt_word+'(?!\w)')
    trigger_word = random.choice(trigger)
    if position == "f":
        sub_word = "_".join([trigger_word, tgt_word])
    elif position == "l":
        sub_word = "_".join([tgt_word, trigger_word])
    elif position == "r":
        word_tokens = tgt_word.split("_")
        word_tokens = [i for i in word_tokens if len(i) > 0]
        random_index = random.randint(0, len(word_tokens))
        word_tokens.insert(random_index, trigger_word)
        sub_word = "_".join(word_tokens)
    pert_code = pattern.sub(sub_word, code)
    return pert_code, 1
