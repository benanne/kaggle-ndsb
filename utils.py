import time
import platform
import numpy as np 
import gzip


def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())
    
def hostname():
    return platform.node()
    
def generate_expid(arch_name):
    return "%s-%s-%s" % (arch_name, hostname(), timestamp())



def one_hot(vec, m=None):
    if m is None:
        m = int(np.max(vec)) + 1

    return np.eye(m)[vec]




def log_losses(y, t, eps=1e-15):
    if t.ndim == 1:
        t = one_hot(t)

    y = np.clip(y, eps, 1 - eps)
    losses = -np.sum(t * np.log(y), axis=1)
    return losses

def log_loss(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    losses = log_losses(y, t, eps)
    return np.mean(losses)

def accuracy(y, t):
    if t.ndim == 2:
        t = np.argmax(t, axis=1)
        
    predictions = np.argmax(y, axis=1)
    return np.mean(predictions == t)

def softmax(x): 
    m = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=1, keepdims=True)

def entropy(x):
    h = -x * np.log(x)
    h[np.invert(np.isfinite(h))] = 0
    return h.sum(1)



def conf_matrix(p, t, num_classes):
    if p.ndim == 1:
        p = one_hot(p, num_classes)
    if t.ndim == 1:
        t = one_hot(t, num_classes)
    return np.dot(p.T, t)


def accuracy_topn(y, t, n=5):
    if t.ndim == 2:
        t = np.argmax(t, axis=1)
    
    predictions = np.argsort(y, axis=1)[:, -n:]    
    
    accs = np.any(predictions == t[:, None], axis=1)

    return np.mean(accs)


def current_learning_rate(schedule, idx):
    s = schedule.keys()
    s.sort()
    current_lr = schedule[0]
    for i in s:
        if idx >= i:
            current_lr = schedule[i]

    return current_lr


def load_gz(path): # load a .npy.gz file
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
        return np.load(f)
    else:
        return np.load(path)


def log_loss_std(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    losses = log_losses(y, t, eps)
    return np.std(losses)
