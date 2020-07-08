import unicodedata, collections, six
from .bert import TOKEN_CLS, TOKEN_SEP, TOKEN_UNK

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
          return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
          return text
        else:
          raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    fp = open(vocab_file, 'rb')
    for line in fp:
        token = convert_to_unicode(line.strip())
        vocab[token] = index
        index += 1
    return vocab

class Tokenizer(object):

    def __init__(self,
                 token_dict,
                 pad_index=0,
                 cased=False):
        """Initialize tokenizer.

        :param token_dict: A dict maps tokens to indices.
        :param token_cls: The token represents classification.
        :param token_sep: The token represents separator.
        :param token_unk: The token represents unknown token.
        :param pad_index: The index to pad.
        :param cased: Whether to keep the case.
        """
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        self._token_cls = TOKEN_CLS
        self._token_sep = TOKEN_SEP
        self._token_unk = TOKEN_UNK
        self._pad_index = pad_index
        self._cased = cased

    @classmethod
    def from_file(cls, vocab_path, pad_index=0, cased=False):
        token_dict = load_vocab(vocab_path)
        return cls(token_dict, pad_index, cased)
        

    @staticmethod
    def _truncate(first_tokens, second_tokens=None, max_len=None):
        if max_len is None:
            return

        if second_tokens is not None:
            while True:
                total_len = len(first_tokens) + len(second_tokens)
                if total_len <= max_len - 3:  # 3 for [CLS] .. tokens_a .. [SEP] .. tokens_b [SEP]
                    break
                if len(first_tokens) > len(second_tokens):
                    first_tokens.pop()
                else:
                    second_tokens.pop()
        else:
            del first_tokens[max_len - 2:]  # 2 for [CLS] .. tokens .. [SEP]

    def _pack(self, first_tokens, second_tokens=None):
        first_packed_tokens = [self._token_cls] + first_tokens + [self._token_sep]
        if second_tokens is not None:
            second_packed_tokens = second_tokens + [self._token_sep]
            return first_packed_tokens + second_packed_tokens, len(first_packed_tokens), len(second_packed_tokens)
        else:
            return first_packed_tokens, len(first_packed_tokens), 0

    def _convert_tokens_to_ids(self, tokens):
        unk_id = self._token_dict.get(self._token_unk)
        return [self._token_dict.get(token, unk_id) for token in tokens]

    def tokenize(self, first, second=None):
        first_tokens = self._tokenize(first)
        second_tokens = self._tokenize(second) if second is not None else None
        tokens, _, _ = self._pack(first_tokens, second_tokens)
        return tokens

    def encode(self, first, second=None, max_len=None):
        first_tokens = self._tokenize(first)
        second_tokens = self._tokenize(second) if second is not None else None
        self._truncate(first_tokens, second_tokens, max_len)
        tokens, first_len, second_len = self._pack(first_tokens, second_tokens)

        token_ids = self._convert_tokens_to_ids(tokens)
        segment_ids = [0] * first_len + [1] * second_len

        if max_len is not None:
            pad_len = max_len - first_len - second_len
            token_ids += [self._pad_index] * pad_len
            segment_ids += [0] * pad_len

        return token_ids, segment_ids

    def decode(self, ids):
        sep = ids.index(self._token_dict[self._token_sep])
        try:
            stop = ids.index(self._pad_index)
        except ValueError as e:
            stop = len(ids)
        tokens = [self._token_dict_inv[i] for i in ids]
        first = tokens[1:sep]
        if sep < stop - 1:
            print(tokens, sep, stop)
            second = tokens[sep + 1:stop - 1]
            return first, second
        return first

    def _tokenize(self, text):
        text = convert_to_unicode(text)
        if not self._cased:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
        return tokens

    def _word_piece_tokenize(self, word):
        if word in self._token_dict:
            return [word]
        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop
        return tokens

    @staticmethod
    def _is_punctuation(ch):
        code = ord(ch)
        return 33 <= code <= 47 or \
            58 <= code <= 64 or \
            91 <= code <= 96 or \
            123 <= code <= 126 or \
            unicodedata.category(ch).startswith('P')

    @staticmethod
    def _is_cjk_character(ch):
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_space(ch):
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
            unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_control(ch):
        return unicodedata.category(ch) in ('Cc', 'Cf')

    def rematch(self, text, tokens):
        """Try to find the indices of tokens in the original text.
        """
        text = convert_to_unicode(text)
        if not self._cased:
            new_text = ''
            match = []
            for idx, ch in enumerate(text):
                sub = unicodedata.normalize('NFD', ch)
                for offset, c in enumerate(sub):
                    if unicodedata.category(c) != 'Mn':
                        if ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                            continue
                        new_text += c
                        match.append(idx)
            new_text = new_text.lower()
        else:
            new_text = text
            match = list(range(len(text)))
        prefix = 0
        bounds = []
        for token in tokens:
            if token.startswith('##'):
                token = token[2:]
            sub_text = new_text[prefix:]
            token_len = len(token)
            start_indice = sub_text.find(token)
            if start_indice > -1:
                start = prefix + start_indice
                end = start + token_len
                bounds.append((start, end))
                prefix = end
            else:
                raise Exception('Token "{}" not found!'.format(token))
        intervals = [(match[bound[0]], match[bound[1] - 1] + 1) for bound in bounds]
        return intervals

    def transform_bound(self, intervals, start, end):
        """transform text sub-string bound indices to tokens bound indices.
        
        Arguments:
            intervals {list} -- list of (start_indice, end_indice)
            start {int} -- start indice of text
            end {int} -- end indice of text, note that the end indice = char index + 1
        
        Returns:
            tuple -- token start index and end index, end index is the exact index(not +1)
        """
        for i, (s, e) in enumerate(intervals):
            if i == 0:
                last_end = 0
            else:
                last_end = intervals[i - 1][-1]
            if i == len(intervals) - 1:
                next_start = 1e7
            else:
                next_start = intervals[i + 1][0]
                
            if last_end <= start < end:
                token_start = i
            if s < end <= next_start:
                token_end = i
        return token_start, token_end
