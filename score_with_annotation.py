from collections import defaultdict
import json, codecs
import csv
import sys

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import numpy as np

def score_with_annotations(annotations_filename, submission_filename, labels_filename, out_file, LABELS=None):
	'''
	A quick and dirty script to score model output both for overall accuracy and for accuracy on each tag from
	our reference annotations.

	Inputs:
		annotations file: This should be one of the two *annotations.txt files included with this script.
		submission file: This should be your models output in csv format, with each line consisting of a "pairID,predicted_label" pair.
		data file: This should be the jsonl version of a MultiNLI dev set file. It is used to extract the correct labels.
	'''
	annotations = {}
	all_annot=set()
	if annotations_filename:
		with codecs.open(annotations_filename, 'r', encoding="utf-8") as f:
			for line in f:
				s = line.strip().split('\t')
				annotations[s[0]] = s[1:]
				all_annot.update(s[1:])

	labels = {}
	with codecs.open(labels_filename, 'r', encoding="utf-8") as f:
		lines=[line for line in f.readlines() if len(line.strip())>0]
		for line in lines:
			ex = json.loads(line)
			if ex['gold_label'] in LABELS.values() or ex['gold_label']=='hidden': #== '-':
				labels[ex['pairID']] = ex['gold_label']
	sentids=set(labels.keys())
	preds={}
	preds_labels_annotation = {ann:{'preds':[], 'labels':[]} for ann in all_annot}
	with codecs.open(submission_filename, 'r', encoding="utf-8") as f:
		lines=[line.strip().split(',') for line in f.readlines() if len(line.strip())>0]
		preds={s[0]:s[1] for s in lines[1:] }
	for pairID in sentids:
		if pairID in annotations.keys():
			for annotation in annotations[pairID]:
				preds_labels_annotation[annotation]['preds'].append(preds[pairID])
				preds_labels_annotation[annotation]['labels'].append(labels[pairID])

	with codecs.open(out_file, "w", encoding="utf-8") as f:
		out_dict = {annotation[:]:str(accuracy_score(preds_labels_annotation[annotation]['labels'],preds_labels_annotation[annotation]['preds'])) for annotation in preds_labels_annotation.keys()}
		ids_sorted=sorted(preds.keys())
		p=[preds[k] for k in ids_sorted]
		l=[labels[k] for k in sorted(labels.keys())]
		out_dict["Accuracy"] = accuracy_score(l,p)
		fscore=precision_recall_fscore_support(l,p)
		#print fscore
		for x in range(len(LABELS.keys())):
			if len(fscore[0])>x:
				out_dict['prec-'+LABELS[x]]=list(fscore[0])[x]
				out_dict['recall-'+LABELS[x]]=list(fscore[1])[x]
				out_dict['fscore-'+LABELS[x]]=list(fscore[2])[x]
		w = csv.DictWriter(f, sorted(out_dict.keys()), delimiter="\t")
		w.writeheader()
		w.writerow(out_dict)
		#creport=classification_report(l,p, target_names=[LABELS[x] for x in xrange(len(LABELS.keys()))])
		#f.write('\n\n'+creport) 

if __name__ == "__main__":
	if len(sys.argv) < 4:
		print("Usage: score_with_annotations.py matched_annotations.txt PATH_TO_matched_submission.csv PATH_TO_matched.jsonl")

	
	annotations_filename = sys.argv[1]
	submission_filename = sys.argv[2]
	labels_filename = sys.argv[3]

	score_with_annotations(annotations_filename, submission_filename, labels_filename, out_file='trial_score.txt')
	
