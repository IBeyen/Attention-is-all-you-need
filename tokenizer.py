from tqdm import tqdm

def get_UTF8(text):
    tokens = text.encode('utf-8')
    tokens = list(map(int, tokens)) # Converts string to list with tokens in [0, 255]
    return tokens

"""
Takes a list of tokens and returns the number of each pair of neighboring tokens

Parameters
------------
tokens_list: List[int]
    List of tokenized string
    
Returns
------------
Dictionary [(int, int) -> int]
    map from pairs to counts
"""
def get_pairs(tokens_list):
    pair_counts = {} # pair -> count
    for i in range(len(tokens_list)-1):
        pair = (tokens_list[i], tokens_list[i+1])
        pair_counts[pair] = pair_counts.get(pair, 0)+1
    return pair_counts

""" 
Takes a list of tokens and and merges all occurrences of a given pair to a single new token

Parameters
------------
tokens_list: List[int]
    Tokenized string
pair: (int, int)
    pair of tokens to find and replace
new_token: int
    new token to replace pair with
    
Returns
------------
List[int]
    A copy of tokens_list with all occurrences of pair replaced with new_token
"""
def token_merge(tokens_list, pair, new_token):
    new_tokens_list = tokens_list.copy()
    for i in reversed(range(len(new_tokens_list)-1)):
        if pair == (new_tokens_list[i], new_tokens_list[i+1]):
            new_tokens_list[i] = new_token
            new_tokens_list.pop(i+1)
    return new_tokens_list

def train_tokenizer(string, vocab_size):
    vocab = {} # idx -> byte string
    merges = {} # pair -> idx
    for i in range(256):
        vocab[i+3] = bytes([i])
    tokens = get_UTF8(string)
    for idx in tqdm(range(259, vocab_size)):
        pairs = get_pairs(tokens)
        most_pair = max(pairs, key=pairs.get)
        merges[most_pair] = idx
        vocab[idx] = vocab[most_pair[0]] + vocab[most_pair[1]]
        tokens = token_merge(tokens, most_pair, idx)
        
    return merges, vocab

def encode_tokens(string, merges):
    tokens = get_UTF8(string)
    while True:
        pairs = get_pairs(tokens)
        pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        tokens = token_merge(tokens, pair, merges[pair])
    return tokens
        
def decode_tokens(tokens, vocab):
    byte_string = b""
    for token in tokens:
        byte_string += vocab[token]
    return byte_string.decode("utf-8")