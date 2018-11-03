from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import merge, Lambda, Bidirectional, CuDNNLSTM, Dense, TimeDistributed, Reshape
from keras.layers.normalization import BatchNormalization
import tensorflow as tf


class ScoreAlign(Layer):
    def __init__(self, hidden_dim, output_dim, keep_dim=True, **kwargs):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
        super(ScoreAlign, self).__init__(**kwargs)

    def build(self, input_shape):
        a_shape, b_shape = input_shape
        concat_dim = a_shape[-1] * 2 # ~ 1200
        self.n = a_shape[1]

        self.FF     = TimeDistributed(Dense(self.hidden_dim))
        self.FF_out = TimeDistributed(Dense(self.output_dim))
        
         # if output_dim is over 1, then we can't discard dimensionality
        assert not keep_dim and output_dim == 1

        self.reshape    = Reshape((self.n**2, concat_dim))
        self.rereshape  = Reshape((self.n,self.n)) if squash else Reshape((self.n,self.n,self.output_dim))
        super(ScoreAlign, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        a, b = x
        n = self.n

        # til_a :: (?, n, n, 300)
        # til_b :: (?, n, n, 300)
        tile_a = K.tile(K.expand_dims(a, 2), [1, 1, n, 1])
        tile_b = K.tile(K.expand_dims(b, 1), [1, n, 1, 1])

        # til_a :: (?, n, n, 1, 300)
        # til_b :: (?, n, n, 1, 300)
        tile_a = K.expand_dims(tile_a, 3)
        tile_b = K.expand_dims(tile_b, 3)

        # only exists for my sanity. no need to construct new tensor
        # alignment :: (?, n, n , 2, 300)
        alignment = K.concatenate([tile_a,tile_b], axis=3)

        # :: (?, n, n, 300)
        minus_component     = alignment[:,:,:,0,:] - alignment[:,:,:,1,:]
        # :: (?, n, n, 300)
        dot_component       = alignment[:,:,:,0,:] * alignment[:,:,:,1,:]
        # :: (?, n, n, 600)
        # concat_component    = K.concatenate([alignment[:,:,:,0,:], alignment[:,:,:,1,:]])

        # :: [ (?, n, n, 300), (?, n, n, 300), (?, n, n, 600) ]
        # components = [dot_component, concat_component, minus_component]
        components = [dot_component, minus_component]

        # :: (?, n, n, 1200)
        super_alignment = K.concatenate(components, axis=-1)

        # :: (?, n**2, 1200)
        super_alignment = self.reshape(super_alignment)
        
        # :: (?, n**2, self.output_dim)
        scores = self.FF(super_alignment)

        # :: (?, n**2, 1)
        scores = self.FF_out(scores)

        # :: (?, n, n)
        return self.rereshape(scores)

    def compute_output_shape(self, input_shape):
        a_shape, b_shape = input_shape
        return (a_shape[0], a_shape[1], b_shape[1])

class Align(Layer):
    def __init__(self, normalize=True, **kwargs):
        self.normalize = normalize
        super(Align, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Align, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        a, b = x
        # if we normalize, this layer outputs the cosine similarity
        if self.normalize:
            a = K.l2_normalize(a, axis=2)
            b = K.l2_normalize(b, axis=2)

        return K.batch_dot(a, b, axes=[2, 2])

    def compute_output_shape(self, input_shape):
        a_shape, b_shape = input_shape
        return (a_shape[0], a_shape[1], b_shape[1])

def _vectoralign(a,b,n=42):
    def _align_vectors(x):
        a, b = x

        tile_a = K.tile(K.expand_dims(a, 2), [1, 1, n, 1])
        tile_b = K.tile(K.expand_dims(b, 1), [1, n, 1, 1])

        # minus_component = tile_a - tile_b
        dot_component = tile_a * tile_b
        # concat_component    = K.concatenate([tile_a, tile_b], axis=-1)
        return dot_component
        # return K.concatenate([dot_component, minus_component], axis=-1)

    return Lambda(_align_vectors)([a, b])


def _dotalign(a, b, normalize):
    return Align(normalize)([a,b])

def _ffalign(a, b, hidden_dim, output_dim, keep_dim=False):
    return ScoreAlign(hidden_dim, output_dim, keep_dim)([a,b])

def _softalign(sentence, alignment, transpose=False):
    def _normalize_attention(attmat):
        att = attmat[0]
        mat = attmat[1]
        if transpose:
            att = K.permute_dimensions(att,(0, 2, 1))
        # 3d softmax
        e = K.exp(att - K.max(att, axis=-1, keepdims=True))
        s = K.sum(e, axis=-1, keepdims=True)
        sm_att = e / s
        return K.batch_dot(sm_att, mat)

    return Lambda(_normalize_attention)([alignment, sentence])