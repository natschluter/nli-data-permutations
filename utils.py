import os, operator
import json, codecs, random
import numpy as np

from os.path import basename
from collections import Counter

def getMeanEmbedding(covered_words):
  #calculating average of 1000 least frequent
  covered_words.sort(key=operator.itemgetter(0))
  least1000=[x[1] for x in covered_words[:1000]]
  ave_vec=np.average(least1000,axis=0)
  return ave_vec


def get_embedding_matrix(fn, vocab_size, vocab_dim, tokenizer, unknown_is_mean, start_end_is_mean):
  filebase_name = ".".join(basename(fn).split(".")[:-1])
  s=''
  if unknown_is_mean:
    s+='UNKmean_'
  if start_end_is_mean:
    s+='SEmean_'
  EMB_STORE = s+filebase_name+'_'+str(vocab_size)+'_words.weights'.format(filebase_name,vocab_size)
  if os.path.exists(EMB_STORE + '.npy'):
      embedding_matrix = np.load(EMB_STORE + '.npy')
      return embedding_matrix, len(embedding_matrix[0])

  embeddings_index = {}
  f = codecs.open(fn, "r", encoding="utf-8")
  values=[]
  for line in f:
    line = line.rstrip()
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
  f.close()

  VOCAB_DIM=len(values)-1
  print(VOCAB_DIM)
  # prepare embedding matrix
  missing_word_count=0
  
  unknown_words=[] #to be list of indices for embedding_matrix
  covered_words=[]
  word_freqs=tokenizer.word_counts
  start_end_words=[]; start_end_tags=["<BOS>","<EOS>"]
  embedding_matrix = np.zeros((vocab_size, VOCAB_DIM))
  for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if word in start_end_tags:
      start_end_words.append(i)
    elif embedding_vector is not None:
      # words not found in embedding index will be all-zeros.
      embedding_matrix[i] = embedding_vector
      covered_words.append((word_freqs[word],embedding_vector))
    else:
      unknown_words.append(i)
      missing_word_count+=1
    #  print('Missing from embedding: {}'.format(word))
 
  if unknown_is_mean or start_end_is_mean:
    mean_least_freq_embedding=getMeanEmbedding(covered_words)
  #unknown words get mean vector  rather than 0s
  if unknown_is_mean:
      for i in unknown_words:
        embedding_matrix[i]=np.copy(mean_least_freq_embedding)
  if start_end_is_mean:
      for i in start_end_words:
        embedding_matrix[i]=np.copy(mean_least_freq_embedding)


  print("Coverage ", (vocab_size+missing_word_count)/vocab_size, 'missing', missing_word_count, 'out of', vocab_size)
  np.save(EMB_STORE, embedding_matrix)
  return embedding_matrix, VOCAB_DIM

# there is a wee bit heuristics behind this impl.
#   1. words that do not occur in the a list of filtered words, are skipped
#   2. We do not load in empty vectors. This happened to some embeddings
def read_embedding_to_dic(filename, filter_words=None):
  w2i = {}
  for line in open(filename, 'r', encoding="utf-8"):
    words = line.lower().strip().split(' ')
    if filter_words and words[0] not in filter_words: 
      continue
    w2i[words[0]] = np.array(words[1:], dtype=np.float32)
  return w2i

def read_lexicon(filename):
  lexicon = {}
  for line in open(filename, 'r', encoding="utf-8"):      
    words = line.lower().strip().split()
    lexicon[words[0]] = words[1:]
  return lexicon

def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None, skip_neutral=False): #, subsample=1.0):
  # infile=open(fn)
  # content=infile.read()
  # lines=[x+'}' for x in content.strip().split('}') if len(x)>0]
 #random.seed(9001)
 for i, line in enumerate(open(fn)):
  # for line in enumerate(lines):
    if limit and i > limit: break
    if len(line.strip())==0: break

    data = json.loads(line)
    label = data['gold_label']
    s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
    s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
    # s1 = data['sentence1'].replace("(", " ( ").replace(")", " ) ")
    # s2 = data['sentence2'].replace("(", " ( ").replace(")", " ) ")
 #   if random.random()<1.0-subsample and label=='entailment': continue
    if skip_neutral and label == "neutral": continue
    if skip_no_majority and label == '-': continue

    yield (label, s1, s2)

def get_data(fn, limit=None, skip_neutral=False):#, subsample=1.0):
  from keras.utils import np_utils
  
  raw_data = list(yield_examples(fn=fn, limit=limit, skip_neutral=skip_neutral))#, subsample=subsample))
  #left = ["<BOS> "+ s1+" <EOS>" for _, s1, s2 in raw_data]
  #right = ["<BOS> "+ s2+" <EOS>" for _, s1, s2 in raw_data]
  left = [s1 for _, s1, s2 in raw_data]
  right = [s2 for _, s1, s2 in raw_data]
  #print("left sentence length in {0}: {1}".format(fn.split("/")[-1], max(len(x.split()) for x in left)))
  #print("right sentence length in {0}: {1}".format(fn.split("/")[-1], max(len(x.split()) for x in right)))

  if skip_neutral:
    LABELS = {'contradiction': 0, 'entailment': 1, 'hidden': 1}
  else:
    LABELS = {'contradiction': 0, 'entailment': 1, 'neutral': 2, 'hidden': 1}
  Y = np.array([LABELS[l] for l, s1, s2 in raw_data if l in LABELS.keys()])
  Y = np_utils.to_categorical(Y, len(LABELS)-1)

  return left, right, Y

def get_data_nolabels(fn, limit=None, skip_neutral=False):
  from keras.utils import np_utils

  raw_data = list(yield_examples(fn=fn, limit=limit, skip_neutral=skip_neutral))
  s=[]
  for _, s1, s2 in raw_data:
    s.append("<BOS> "+ s1+' '+s2+" <EOS>")
  return s

def parse_optimizer(string_input):
  # motivation for tf optimizers https://github.com/fchollet/keras/issues/1021#issuecomment-270566219
  if string_input.startswith("tf."):
    from keras.optimizers import TFOptimizer
    import tensorflow as tf

    print("> using tensorflow optimizer")
    if string_input == "tf.sgd":
      return TFOptimizer(tf.train.GradientDescentOptimizer(learning_rate=0.1))

    if string_input == "tf.rmsprop":
      # lr taken from keras standard impl
      return TFOptimizer(tf.train.RMSPropOptimizer(learning_rate=0.001))
  else:
    return string_input
