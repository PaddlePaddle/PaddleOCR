class Converter(object):
    def __init__(self, dict_path):
        self.chars = open(dict_path, 'r').read().splitlines()  
        self.chr2idx = {char: i for i, char in enumerate(self.chars)}  
    
    def encode(self, str):
        return [self.chr2idx.get(char) for char in str]
    
    def decode(self, indices):
        chars = [self.chars[int(idx)] for idx in indices]
        return ''.join(chars)