import os, codecs, sys, logging, datetime
import json, tempfile, argparse, timeit

from keras.models import Model
from keras.utils import np_utils
from keras.layers import Input, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
import keras.preprocessing.text
import os.path

#np.random.seed(1337)
#import tensorflow as tf
#tf.set_random_seed(1337)

from utils import get_embedding_matrix, get_data, parse_optimizer, get_data_nolabels
from model import construct_model

def check_configuration_file():
	test_sets = json.load(open(args.testsets, "r"))
	for _, filenames in test_sets.items():
		for fname in filenames:
			if fname != None and not os.path.isfile(fname):
				print(fname+' does not exist')

def make_experiment_name(args):
	s=args.dataset.upper()
	if s == 'MULTI':
		if args.matched:
			s+='_MA'
		else:
			s+='_MISMA'
	if args.scrambled:
		s+='_SCRAM'
	if args.enable_projection:
		s+='_EP'
		s+='-'+args.projection_mode
	if args.dot_alignment:
		s+='_DOT'
	s+='_DL'+str(args.dense_layers)
	s+='_sum'+args.sum_mode
	return s

def save_model(model, out_folder):
	# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
	# serialize model to JSON
	model_json = model.to_json()
	with open(out_folder+"model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(out_folder+"model.h5")
	print("Saved model to disk")

def save_embedding(embedding, path):
	print("Retrieving embedding weights")
	trained_embedding = np.array(embedding.get_weights())
	trained_embedding = trained_embedding.reshape(VOCAB, SENT_HIDDEN_SIZE)

	print("writing post training embeddings to files")
	with codecs.open(path, "w", encoding="utf-8") as f_post:
		for word, i in tokenizer.word_index.items():
			line = "{0} {1}\n".format(word, " ".join(map(str, trained_embedding[i])))

	print("done writing embeddings to file")

def get_vocabulary():
	print('Loading vocab...')
	dataset_folder='/home/natalie/data/'
	text=[]

	#load multi-nli dev/test, matched/mismatched set
	files=['multinli_0.9_test_matched_unlabeled.jsonl', 'multinli_0.9_test_mismatched_unlabeled.jsonl', 'multinli_1.0_dev_matched.jsonl', 'multinli_1.0_dev_mismatched.jsonl']
	for f in files:
		print(f)
		text.extend(get_data_nolabels(dataset_folder+'multinli_1.0/'+f))
	
	#load snli dev/test set
	snlifolder=dataset_folder+'snli_1.0/snli_1.0/'
	print(snlifolder)
	text.extend(get_data_nolabels(snlifolder+'snli_1.0_dev.jsonl'))
	text.extend(get_data_nolabels(snlifolder+'snli_1.0_test.jsonl'))

	#load demarneffe test set
	print('demarneffe')
	text.extend(get_data_nolabels('/home/djam/data/marneffe/marneffe_test.jsonl'))
	return text

def get_best_model_on_accuracy():
	epoch_files=[file for file in os.listdir(epochs_folder) if file.startswith('weights')]
	max_accuracy=float(epoch_files[0][:-5].split('-')[-1].split('_')[1])
	filename_max_accuracy=epoch_files[0]
	for fname in epoch_files[1:]:
		temp_acc=float(fname[:-5].split('-')[-1].split('_')[1])
		if temp_acc>max_accuracy:
			max_accuracy=temp_acc
			filename_max_accuracy=fname
	return filename_max_accuracy, max_accuracy

def parse_arguments():	
	parser = argparse.ArgumentParser()
	parser.add_argument('--pref', help="extra prefix for experiment name", default='')
	parser.add_argument('--matched', help="matched/mismatched for multi", action="store_true", default=False)
	parser.add_argument('--scrambled', help="scrambled train set", action="store_true", default=False)
	parser.add_argument('--dataset', required=True, help="snli or multi")
	parser.add_argument('--train', required=True, help="train file. JSONL file plz")
	parser.add_argument('--dev', required=True, help="dev file. JSONL file plz")
	parser.add_argument('--test', required=True, help="snli test file. JSONL file plz")
	parser.add_argument('--test2', required=False, help="2nd nli test file. JSONL file plz", default=False)
	parser.add_argument('--embedding', required=True, help="embedding file")
	parser.add_argument('--fixed_embedding', action="store_true", help="fixed embedding", default=False)
	parser.add_argument('--enable_projection', action="store_true", help="initial projection layer", default=False)
	parser.add_argument('--projection_mode', required=False)
	parser.add_argument('--dot_alignment', action='store_true', default=False, help='dot if present, otherwise we do ff alignment')
	parser.add_argument('--optimizer', required=False, help="optimizer", default='rmsprop')
	parser.add_argument('--binary', action='store_true', help="enable binary classification", default=False)
	parser.add_argument('--sum_mode', help="which approach. sum vs lstm")
	parser.add_argument('--dense_layers', required=True, type=int, help="number of dense layers before final layer")
	parser.add_argument("--result_file", required=False, default='results.txt')
	parser.add_argument("--no_save_model", action="store_false", required=False, default=True)
	parser.add_argument("--out_folder", required=False, default='score_folder/')
	parser.add_argument("--batch_size", required=False, type=int, default=512)
	parser.add_argument("--epochs", required=False, type=int)
	parser.add_argument("--testsets", required=False, default='configuration/server_config.json')
	parser.add_argument("--do_all_tests", required=False, action="store_true", default=False, help='do all tests')
	parser.add_argument("--unknown_is_mean", action="store_true", required=False, default=False)
	parser.add_argument("--start_end_is_mean", action="store_true", required=False, default=False)
	parser.add_argument("--limit", required=False, type=int, default=None, help='limit training examples for development')
	parser.add_argument("--save_all_epochs", required=False, action="store_true", default=False, help='store model from each epoch')
	#parser.add_argument("--class_weight", required=False, default='1-1-1', help="class weights for training as string 1-1-1=C-E-N is default")
	#parser.add_argument("--subsample_entail", required=False, default='1.0', type=float, help="random subsample rate to filter entailments from training")

	args = parser.parse_args(sys.argv[1:])
	return args

args=parse_arguments()

if args.dataset not in ['snli', 'multi']:
	print('ERROR for dataset arg', args.dataset)
	sys.exit()
#cw=args.class_weight.split('-')
#class_weight={x:float(cw[x]) for x in range(len(cw))}

#check_configuration_file()
expername = make_experiment_name(args)
model_name = expername

start = timeit.default_timer()

VERBOSE = 1

print("> SETTINGS")
print("OPTIMIZER:			   {0}".format(args.optimizer))
print("BATCH SIZE:			   {0}".format(args.batch_size))
print("BINARY CLASSIFICATION:   {0}".format(args.binary))
print("DENSE LAYERS:			{0}".format(args.dense_layers))
print("DOT ALIGN ENABLED		{0}".format(args.dot_alignment))
print("PROJECTION MODE:		 {0}".format(args.projection_mode))
print("FIXED EMBEDDINGS:		{0}".format(args.fixed_embedding))
print("SUM MODE				 {0}".format(args.sum_mode))
print("TRAINING FILE			{0}".format(args.train))

training 	= get_data(args.train, args.limit, skip_neutral=args.binary)#, subsample=args.subsample_entail)
validation 	= get_data(args.dev, skip_neutral=args.binary)
test 		= get_data(args.test, skip_neutral=args.binary)
#other validation sets
extra_text=get_vocabulary()

tokenizer = Tokenizer(lower=False, filters='')
#tokenizer.fit_on_texts(training[0] + training[1])
tokenizer.fit_on_texts(training[0] + training[1]+extra_text)

# Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
VOCAB = len(tokenizer.word_counts) + 1
EMBED_HIDDEN_SIZE = 300
SENT_HIDDEN_SIZE = 300
BATCH_SIZE = 512 if not args.batch_size else args.batch_size
PATIENCE = 4 # 8
MAX_EPOCHS = 42 if not args.epochs else args.epochs
MAX_LEN = 42
DP = 0.2
L2 = 4e-6
ACTIVATION = "relu"

# read in embedding and translate
print("> fetching word embedding")
embedding_matrix, EMBED_HIDDEN_SIZE = get_embedding_matrix(args.embedding, VOCAB, EMBED_HIDDEN_SIZE, tokenizer, args.unknown_is_mean, args.start_end_is_mean)
embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False if args.fixed_embedding else True)

# OPTIMIZER = 'rmsprop'
# motivation https://github.com/fchollet/keras/issues/1021#issuecomment-270566219
# OPTIMIZER = TFOptimizer(tf.train.GradientDescentOptimizer(0.1))
# OPTIMIZER = TFOptimizer(tf.train.RMSPropOptimizer(0.001)) # doesn't work
OPTIMIZER = parse_optimizer(args.optimizer)

to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

training = prepare_data(training)
validation = prepare_data(validation)
test = prepare_data(test)

premise = Input(shape=(MAX_LEN,), dtype='int32')
hypothesis = Input(shape=(MAX_LEN,), dtype='int32')

# embedding sentences
prem = embed(premise)
hypo = embed(hypothesis)

stope = timeit.default_timer()
timee = (stope - start)/60

print(" Embeddings loaded in {:.2f} mins".format(timee))

# construct model
pred = construct_model(prem, hypo, 
					   SENT_HIDDEN_SIZE, ACTIVATION, L2, DP,
					   args.enable_projection,
					   projection_mode=args.projection_mode,
					   dot_align=args.dot_alignment,
					   sum_strategy=args.sum_mode,
					   dense_layers=args.dense_layers,
					   outlayer_count=2 if args.binary else 3)


model = Model(inputs=[premise, hypothesis], outputs=pred)
#model.compile(optimizer=OPTIMIZER, loss=['categorical_crossentropy','mean_squared_logarithmic_error'], metrics=['accuracy'])
model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print("Training")
_, tmpfn = tempfile.mkstemp()

out_folder=args.out_folder
if out_folder[-1] !='/':
	out_folder+=args.pref+'/'
else:
	out_folder=out_folder[:-1]+args.pref+'/'
out_dir = out_folder+model_name+"/"
if not os.path.exists(out_dir):
	os.makedirs(out_dir)
epochs_folder=out_folder+model_name+'_epochs/'
if args.save_all_epochs:
	if not os.path.exists(epochs_folder):
		os.makedirs(epochs_folder)
	
	
# Save the best model during validation and bail out of training early if we're not improving
#early_stop = EarlyStopping(patience=PATIENCE, monitor='val_loss')
early_stop = EarlyStopping(patience=PATIENCE, monitor='val_acc')
filepath='weights.epoch_{epoch:02d}-loss_{val_loss:.4f}-acc_{val_acc:.4f}.hdf5'
if args.save_all_epochs:
	check_point = ModelCheckpoint(epochs_folder+filepath, save_best_only=False, save_weights_only=True)
else:
	check_point = ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)
callbacks = [early_stop, check_point]
metrics = model.fit([training[0], training[1]], training[2], batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks, verbose=VERBOSE)#, class_weight=class_weight)

stop = timeit.default_timer()
time = (stop - start)/60

if args.save_all_epochs:
	bestmodelfile,bestaccuracy=get_best_model_on_accuracy()
	model.load_weights(epochs_folder+bestmodelfile)
else:
	# Restore the best found model during validation
	model.load_weights(tmpfn)

print(" Model trained in {:.2f} mins".format(time))

#across test files
loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

# save model
if not args.no_save_model:
	save_model(model, out_dir)

# write model performance metrics to file
with codecs.open(args.result_file, "a", encoding="utf-8") as result_f:
	now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
	result_f.write("{0},{1},{2},{3}\n".format(now, metrics.history["acc"][-1], metrics.history["val_acc"][-1], acc))
	
# save embeddings to file if not fixed
if not args.fixed_embedding:
	save_embedding(embed, out_folder+'trained_'+args.embedding)

from operator import itemgetter

if args.binary:
	LABELS = {0:'contradiction', 1:'entailment'}
else:
	LABELS = {0:'contradiction', 1:'entailment', 2:'neutral'}
GLABELS=[val for val in LABELS.values()]; GLABELS.append('hidden')

def write_predictions(name, origin_file, predictions, LABELS):
	lines=[json.loads(line) for line in codecs.open(origin_file, "r", encoding="utf-8") ]
	pair_ids = [line["pairID"] for line in lines if line["gold_label"] in GLABELS]
	predictions = [LABELS[np.argmax(p)] for p in predictions]
	with codecs.open(name, "w", encoding="utf-8") as f:
		f.write('pairID,gold_label\n')
		for pair_id, prediction in zip(pair_ids, predictions):
			f.write(str(pair_id)+','+prediction+"\n")
			
def write_predictions_conf(name, conf_name, origin_file, predictions, LABELS):
	lines=[json.loads(line) for line in codecs.open(origin_file, "r", encoding="utf-8") ]
	pair_ids = [line["pairID"] for line in lines if line["gold_label"] in GLABELS]
	gold_labels = [line["gold_label"] for line in lines if line["gold_label"] in GLABELS]
	if 'dev' in origin_file  or 'test' in origin_file:
		sents=[[line["sentence1"], line["sentence2"]] for line in lines if line["gold_label"] in GLABELS]
	predictions_lab = [LABELS[np.argmax(p)] for p in predictions]
	with codecs.open(name, "w", encoding="utf-8") as f:
		f.write('pairID,gold_label\n')
		for pair_id, prediction in zip(pair_ids, predictions_lab):
			f.write(str(pair_id)+','+prediction+"\n")
	if 'dev' in origin_file  or 'test' in origin_file:
		with codecs.open(conf_name, "w", encoding="utf-8") as f:
			neutrals=[]
			f.write('pairID,prediction,gold_label,'+','.join([LABELS[x] for x in range(len(LABELS))]) +'\n')
			for x in range(len(pair_ids)):
				f.write(str(pair_ids[x])+','+predictions_lab[x]+','+gold_labels[x]+','+','.join([str(y) for y in list(predictions[x])])+"\n")
				if gold_labels[x]=='neutral':
					temp=list(predictions[x][:])
					temp.append(pair_ids[x])
					temp.extend(sents[x])
					neutrals.append(temp)
				
		neutrals.sort(key=itemgetter(2))
		print(len(neutrals))
		with codecs.open(name[:-4]+'_neutrals.txt', 'w', encoding='utf-8') as f:
			for x in neutrals:
				f.write('\t'.join([str(s) for s in x])+'\n')
	return  gold_labels, predictions_lab

if args.save_all_epochs:
	#for the number of epochs write out results.
	epoch_files=[file for file in os.listdir(epochs_folder) if file.startswith('weights')]
	tests=[]; devs=[]; test2s=[]; header=''
	for file in epoch_files:
		model.load_weights(epochs_folder+file)
		epoch=file[:-5].split('-')[0].split('_')[-1]
		curr_outdir=epochs_folder+str(epoch)+'_'+args.dataset+'/'
		if not os.path.exists(curr_outdir):
			os.makedirs(curr_outdir)
		#dev
		predictions = model.predict([validation[0],validation[1]])
		l,p=write_predictions_conf(curr_outdir+'dev_predictions.txt', curr_outdir+'dev_predictions_with_conf.txt', origin_file=args.dev, predictions=predictions, LABELS=LABELS)
		loss, acc = model.evaluate([validation[0], validation[1]], validation[2], batch_size=BATCH_SIZE)
		fscore=precision_recall_fscore_support(l,p)
		s=''; header=''
		for x in range(len(LABELS.keys())):
			header+='\tprec-'+LABELS[x]+'\t'+'recall-'+LABELS[x]+'\t'+'fscore-'+LABELS[x]
			if len(fscore[0])>x:
				s+='\t'+str(list(fscore[0])[x])+'\t'+str(list(fscore[1])[x])+'\t'+str(list(fscore[2])[x])
		devs.append(str(epoch)+'\t'+str(loss)+'\t'+str(acc)+s)
		ofile=open(curr_outdir+'dev_score.txt','w')
		ofile.write('loss\tacc'+header+'\n'+str(loss)+'\t'+str(acc)+s+'\n')	
		ofile.close()
		#test
		predictions = model.predict([test[0],test[1]])
		l,p=write_predictions_conf(curr_outdir+'test_predictions.txt', curr_outdir+'test_predictions_with_conf.txt', origin_file=args.test, predictions=predictions, LABELS=LABELS)
		loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
		s=''; header=''
		for x in range(len(LABELS.keys())):
			header+='\tprec-'+LABELS[x]+'\t'+'recall-'+LABELS[x]+'\t'+'fscore-'+LABELS[x]
			if len(fscore[0])>x:
				s+='\t'+str(list(fscore[0])[x])+'\t'+str(list(fscore[1])[x])+'\t'+str(list(fscore[2])[x])
		tests.append(str(epoch)+'\t'+str(loss)+'\t'+str(acc)+s)
		ofile=open(curr_outdir+'test_score.txt','w')
		ofile.write('loss\tacc'+header+'\n'+str(loss)+'\t'+str(acc)+s+'\n')	
		ofile.close()
		#test2
		if args.test2:
			test2 = get_data(args.test2, skip_neutral=args.binary)
			test2 = prepare_data(test2)		
			predictions = model.predict([test2[0],test2[1]])
			l,p=write_predictions_conf(curr_outdir+'test2_predictions.txt', curr_outdir+'test2_predictions_with_conf.txt', origin_file=args.test2, predictions=predictions, LABELS=LABELS)
			loss, acc = model.evaluate([test2[0], test2[1]], test2[2], batch_size=BATCH_SIZE)
			s=''; header=''
			for x in range(len(LABELS.keys())):
				header+='\tprec-'+LABELS[x]+'\t'+'recall-'+LABELS[x]+'\t'+'fscore-'+LABELS[x]
				if len(fscore[0])>x:
					s+='\t'+str(list(fscore[0])[x])+'\t'+str(list(fscore[1])[x])+'\t'+str(list(fscore[2])[x])
			test2s.append(str(epoch)+'\t'+str(loss)+'\t'+str(acc)+s)
			ofile=open(curr_outdir+'test_score.txt','w')
			ofile.write('loss\tacc'+header+'\n'+str(loss)+'\t'+str(acc)+s+'\n')	
			ofile.close()
		os.remove(epochs_folder+file)
	if args.test2:
		with open(epochs_folder+'test2_scores.txt','w') as ofile:
			ofile.write('epoch\tloss\tacc'+header+'\n')	
			ofile.write('\n'.join(sorted(test2s)))
	with open(epochs_folder+'test_scores.txt','w') as ofile:
		ofile.write('epoch\tloss\tacc'+header+'\n')	
		ofile.write('\n'.join(sorted(tests)))
	with open(epochs_folder+'dev_scores.txt','w') as ofile:
		ofile.write('epoch\tloss\tacc'+header+'\n')	
		ofile.write('\n'.join(sorted(devs)))

if args.testsets and args.do_all_tests:
	# write out predictions
	from score_with_annotation import score_with_annotations

	test_sets = json.load(open(args.testsets, "r")) #comment out for kfold
	test_sets['train']=[args.train,None] #comment out for kfold
	#test_sets={'dev':[args.dev,None]} #put this on for kfolds
	test_sets['test']=[args.test,None]
	if args.test2:
		test_sets['test2']=[args.test2,None]
	for name, filenames in test_sets.items():
		prediction_filename = out_dir+name+'_predictions.txt'
		conf_prediction_filename= out_dir+name+'_predictions_with_conf.txt'
		score_filename = out_dir+name+"_scores.txt"
		data_file, annotation_file = filenames
		print(data_file)
		data = get_data(data_file, skip_neutral=args.binary)
		data = prepare_data(data)

		predictions = model.predict([data[0], data[1]])
		write_predictions_conf(prediction_filename, conf_prediction_filename, origin_file=data_file, predictions=predictions, LABELS=LABELS)	
		score_with_annotations(annotation_file, prediction_filename, data_file, score_filename, LABELS)

	
