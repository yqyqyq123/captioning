import nltk
import pickle
import argparse
from collections import Counter
import operator
from pycocotools.coco import COCO
nltk.download('punkt')
nltk.download('wordnet')

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, option, threshold, most_frequent):
    """Build a simple vocabulary wrapper."""
    if option == "stem":
        nl = nltk.stem.PorterStemmer()
    else:
        nl = stem.WordNetLemmatizer()
        
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        if option == "stem":
            tokens = [nl.stem(word) for word in tokens]
        else:
            tokens = [nl.lemmatize(word) for word in tokens]
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

   
    
    #words = [word for word, cnt in counter.items() if cnt >= threshold]
    c = dict(counter)
    
    sorted_x = dict(sorted(c.items(), key=operator.itemgetter(1),reverse=True))
    words = list(sorted_x.keys())[:most_frequent]
    print(len(words))
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')


    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    print("Option: {}".format(args.option))
    if args.option == "stem":
        vocab_path = args.vocab_stem_path
    elif args.option == "lemma":
        vocab_path = args.vocab_lemma_path
    else:
        print("No Such Option")
        return
    vocab = build_vocab(json=args.caption_path, option = args.option, threshold=args.threshold, most_frequent=args.most_frequent)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default="./datasets/coco2014/trainval_coco2014_captions/captions_train2014.json", 
                        help='path for train annotation file')
    parser.add_argument('--vocab_stem_path', type=str, default="vocab_stem.pkl", 
                        help='path for saving stem vocabulary wrapper')
    parser.add_argument('--vocab_lemma_path',type=str, default="vocab_lemma.pkl", 
                        help='path for saving lemma vocabulary wrapper')
    parser.add_argument('--option', type=str, default="lemma", 
                        help='lemma or stem')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    parser.add_argument('--most_frequent', type=int, default=1000, 
                        help='top most frequent words')
    args = parser.parse_args()
    main(args)