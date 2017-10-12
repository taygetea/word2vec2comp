from gensim.models import Word2Vec
import logging
from nltk.corpus import stopwords
import nltk.data
import os
import time
import pandas as pd
import plac
import re
import codecs
from multiprocessing import Process, Pool
from sklearn.decomposition import PCA
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%H:%M:%S')

# Experimental multiprocess tokenization

"""
sentences = []
for document in inputdata:
    sentences += to_sentences(document, tokenizer, remove_stopwords=False)
"""

def multitokenize(inputdata, processes=4):

    def chunks(l, n):
        n = int(len(l)/n)
        for i in range(0, len(l), n):
            yield l[i:i+n]



    pool = Pool(processes=processes)

    partitioned = list(chunks(inputdata, processes))

    tokenized = pool.map(to_sentences, partitioned)
    sentences = []
    for index, section in enumerate(tokenized):
        sentences.extend(section)
    return sentences

# convert a string of text to a list of words
def separate_words(sentence, remove_urls=True, remove_stopwords=False):


    wordlist = re.sub("[^a-zA-Z]", " ", sentence).lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        wordlist = [w for w in wordlist if w not in stops]

    return wordlist


# parse a document to sentences
def to_sentences(document, remove_stopwords=False, remove_urls=False):


    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    sentences = tokenizer.tokenize(document.strip())
    sentences = [separate_words(sentence, remove_stopwords, remove_urls)
                 for sentence in sentences if len(sentence) > 0]
    # import pdb; pdb.set_trace()

    return sentences
# train a model
def train(sentences, features=100, mincount=200, workers=4, context=10,
        sample=1e-3, save=False, precomp=True):
    model = Word2Vec(sentences,
                     sg=1,
                     workers=workers,
                     size=features,
                     min_count=mincount,
                     window=context,
                     sample=sample)
    if precomp:
        model.init_sims(replace=True)
    if save:
        savestrip = "".join(save.split(".")[:-1])
        model.save(os.path.join(savestrip+".model"))
    logging.info("Training complete. Output contains %d words with %d features" %
            (model.wv.syn0.size, model.wv.syn0[0].size))
    return model


def run_pca(df, outfile="components.csv", n_components=9):
    pca = PCA(n_components)
    pc = pca.fit_transform(df)
    component_names = ["PC"+str(x) for x in range(0,len(pc[0]+1))]
    df2 = pd.DataFrame(pc, index=df.index, columns=component_names)
    wordcomponents = pd.DataFrame([df2.sort_values(by=x).index for x in component_names]).transpose()

    def disprow(row):
        if row == 1:
            r = wordcomponents.iloc[:1,0:n_components]
        elif row == -1:
            r = wordcomponents.iloc[-1:,0:n_components]
        r = r.to_string(header=False, index=False).split(" ")
        r = "".join([(x).ljust(15) for x in r if x.strip()])
        return r

    logging.info("".join([x.ljust(15) for x in component_names]))
    logging.info(disprow(1))
    logging.info(disprow(-1))


    wordcomponents.to_csv(outfile)
    logging.info("Components saved at " + os.getcwd() + "/outputs/" + outfile)
    html_table = wordcomponents.to_html(index=False, classes=["centered", "striped"])
    with open(outfile, 'w+') as f:
        f.write(''.join(html_table))
    return wordcomponents



def vec2pca(fname, output, content=None):


    inputdata = pd.Series(codecs.open(fname, "r", "utf-8").readlines()).dropna()

    # sentences = []
    # for document in inputdata:
    #     sentences += to_sentences(document, tokenizer)

    sentences = multitokenize(". ".join(inputdata), processes=4)
    model = train(sentences)

    keys = list(model.wv.vocab.keys())
    df = pd.DataFrame(model[keys], index=keys)


    pcs = run_pca(df, outfile=output)
    return df, pcs


if __name__ == "__main__":
    # vec2pca(filename, outputfile)
    plac.call(vec2pca)
