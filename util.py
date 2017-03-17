import numpy as np
import os
import theano
import theano.tensor as T
import time
import six
from theano.ifelse import ifelse
from collections import OrderedDict
import warnings

logit = lambda u: T.log(u / (1.-u))
logit_np = lambda u: np.log(u / (1.-u)).astype(theano.config.floatX)

#import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams(seed=4884)
'''
def floatX(num):
    if theano.config.floatX == 'float32':
        return numpy.float32(num)
    else:
        raise Exception("{} type not supported".format(theano.config.floatX))
'''


def get_norms(model, gradients):
    """Compute norm of weights and their gradients divided by the number of elements"""
    norms = []
    grad_norms = []
    for param_name, param in model.params.iteritems():
        norm = T.sqrt(T.sum(T.square(param))) / T.prod(param.shape.astype(theano.config.floatX))
        norm.name = 'norm_' + param_name
        norms.append(norm)
        grad = gradients[param]
        grad_norm = T.sqrt(T.sum(T.square(grad))) / T.prod(grad.shape.astype(theano.config.floatX))
        grad_norm.name = 'grad_norm_' + param_name
        grad_norms.append(grad_norm)
    return norms, grad_norms

def create_log_dir(args, model_id):
    model_id += args.suffix + time.strftime('-%y%m%dT%H%M%S')
    model_dir = os.path.join(os.path.expanduser(args.output_dir), model_id)
    os.makedirs(model_dir)
    return model_dir


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
        return [vv for kk, vv in six.iteritems(tparams)]

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


# weight initializer, normal by default
def norm_weight(nin, nout=None, scale=0.001, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype('float32')

floatX = theano.config.floatX
def init_uniform(range=0.1, dtype=floatX):
    return lambda shape: np.random.uniform(-range, range, size=shape).astype(dtype)

def _p(pp, name):
    return '%s_%s' % (pp, name)

def init_zeros(dtype=floatX):
     return lambda shape: np.zeros(shape, dtype=dtype)

def Linear(inp, inp_dim, outp_dim, vs, name="linear", bias=True):
     w = vs.add_param("%s/w" % name, (inp_dim, outp_dim))
     b = vs.add_param("%s/b" % name, (outp_dim,), initializer=init_zeros())
     return inp.dot(w) + b

def norm_clip(dW, max_l2_norm=10.0):
    """
    Clip theano symbolic var dW to have some max l2 norm.
    """
    dW_l2_norm = T.sqrt(T.sum(dW**2.0))
    norm_ratio = (max_l2_norm / dW_l2_norm)
    clip_factor = ifelse(T.lt(norm_ratio, 1.0), norm_ratio, 1.0)
    dW_clipped = dW * clip_factor
    return dW_clipped


def gradient_clipping(grads, tparams, clip_c=1.0):
    g2 = 0.
    for g in grads:
        g2 += (g**2).sum()

    g2 = T.sqrt(g2)
    not_finite = T.or_(T.isnan(g2), T.isinf(g2))
    new_grads = []

    for p, g in zip(tparams.values(), grads):
        new_grads.append(T.switch(g2 > clip_c,
                                       g * (clip_c / g2),
                                       g))

    return new_grads, not_finite, T.lt(clip_c, g2)



def get_param_updates(params=None, grads=None, \
        alpha=None, beta1=None, beta2=None, it_count=None, \
        mom2_init=1e-3, smoothing=1e-6, max_grad_norm=10000.0):
    """
    This update has some extra inputs that aren't used. This is just so it
    can be called interchangeably with "ADAM" updates.
    """

    # make an OrderedDict to hold the updates
    updates = OrderedDict()
    # alpha is a shared array containing the desired learning rate
    lr_t = alpha[0]
    for p in params:
        # get gradient for parameter p
        grad_p = norm_clip(grads[p], max_grad_norm)

        # initialize first-order momentum accumulator
        mom1_ary = 0.0 * p.get_value(borrow=False)
        mom1 = theano.shared(mom1_ary)

        # update momentum accumulator
        mom1_new = (beta1[0] * mom1) + ((1. - beta1[0]) * grad_p)

        # do update
        p_new = p - (lr_t * mom1_new)

        # apply updates to
        updates[p] = p_new
        updates[mom1] = mom1_new
    return updates

#load params
def load_params(path, params):
    pp = np.load(path)
    for kk, vv in six.iteritems(params):
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in six.iteritems(zipped):
        new_params[kk] = vv.get_value()
    return new_params

def save_params(params, filename, symlink=None):
    """Save the parameters.
    Saves the parameters as an ``.npz`` file. It optionally also creates a
    symlink to this archive.
    """
    np.savez(filename, **params)
    if symlink:
        if os.path.lexists(symlink):
            os.remove(symlink)
        os.symlink(filename, symlink)
