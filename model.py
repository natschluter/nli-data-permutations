from keras.layers import Dense, TimeDistributed,Lambda, concatenate, Dropout, LSTM, CuDNNLSTM, Bidirectional, Flatten, Conv2D, Reshape
from keras.layers import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import math

from custom_layers import _dotalign, _softalign, ScoreAlign, _vectoralign

def construct_model(premise, hypothesis,
					SENT_HIDDEN_SIZE, ACTIVATION, L2, DP,
					enable_projection,
					projection_mode,
					dot_align,
					sum_strategy,
					dense_layers,
					outlayer_count):

	prem = premise
	hypo = hypothesis
	#SENT_HIDDEN_SIZE=SENT_HIDDEN_SIZE*2

#	if intra_attention: #motivation for this?

	translate0 = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))
	prem = translate0(prem)
	hypo = translate0(hypo)

	if enable_projection: 
		translate=None
		if projection_mode == "FF": #Further projection
			translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))
		elif projection_mode == "LSTM":
			translate = CuDNNLSTM(SENT_HIDDEN_SIZE, return_sequences=True)
		elif projection_mode == "BiLSTM":
			translate = Bidirectional(CuDNNLSTM(SENT_HIDDEN_SIZE, return_sequences=True))
		else:
			raise ValueError("no corresponding projection mode")

		prem = translate(prem)
		hypo = translate(hypo)
		
	alignment = _dotalign(prem, hypo, normalize=False)

	prem_c = _softalign(prem, alignment, transpose=True)
	hypo_c = _softalign(hypo, alignment)

	minus = Lambda(lambda x: x[0]-x[1])

	prem = concatenate([prem, hypo_c, minus([prem, hypo_c])], axis=-1)
	hypo = concatenate([hypo, prem_c, minus([hypo, prem_c])], axis=-1)

	translate2 = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

	prem = translate2(prem)
	hypo = translate2(hypo)

	aggregator = None
	if sum_strategy == "SUM":
		aggregator = Lambda(lambda x: K.sum(x, axis=1))
	if sum_strategy == "LSTM":
		aggregator = CuDNNLSTM(SENT_HIDDEN_SIZE)
	if sum_strategy == "BiLSTM":
		aggregator = Bidirectional(CuDNNLSTM(SENT_HIDDEN_SIZE))
	if sum_strategy == "CONCAT":
		aggregator =  Flatten()

	assert aggregator is not None

	prem = aggregator(prem)
	hypo = aggregator(hypo)

	joint = concatenate([prem, hypo])
	joint = Dropout(DP)(joint)

	curr_dim=2 * SENT_HIDDEN_SIZE
	for i in range(dense_layers):
		joint = Dense(curr_dim, activation=ACTIVATION, kernel_regularizer=l2(L2))(joint)
		joint = Dropout(DP)(joint)
		joint = BatchNormalization()(joint)
		#curr_dim=int(math.floor(curr_dim/2.0))

	pred = Dense(outlayer_count,  activation='softmax')(joint)
	return pred

	
def construct_conv_model(premise, hypothesis,
						 SENT_HIDDEN_SIZE, ACTIVATION, L2, DP, align_op):
	prem = premise
	hypo = hypothesis

	# highway    
	highway_1 = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))
	highway_2 = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

	# first pass
	prem = highway_1(prem)
	hypo = highway_1(hypo)

	# second pass
	prem = highway_2(prem)
	hypo = highway_2(hypo)

	# alignment
	if align_op == "dot":
		alignment = _dotalign(prem,hypo, normalize=False)
		alignment = Reshape((42,42,1))(alignment)
	elif align_op == "mult":
		alignment = _vectoralign(prem,hypo)
	else:
		raise ValueError("no alignment op")

	# 256, 256, 128 :: capsule baseline
	conv_1 = Conv2D(256, kernel_size=(3,3), activation=ACTIVATION)(alignment)
	conv_2 = Conv2D(256, kernel_size=(3,3), activation=ACTIVATION)(conv_1)
	conv_3 = Conv2D(128, kernel_size=(3,3), activation=ACTIVATION)(conv_2)

	flattened = Flatten()(conv_3)

	# 328, 192
	dense_1 = Dense(328, activation=ACTIVATION)(flattened)
	dense_2 = Dense(192, activation=ACTIVATION)(dense_1)

	dropout = Dropout(DP)(dense_2)

	pred = Dense(3, activation='softmax')(dropout)

	return pred
	
	

