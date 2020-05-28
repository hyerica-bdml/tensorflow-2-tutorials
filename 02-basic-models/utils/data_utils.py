import numpy as np
from tensorflow.keras.datasets.mnist import load_data


def shuffle(x, y):
    """
    x, y를 셔플한다.
    
    Arguments:
    ----------
    x : features 데이터 행렬 (N, ...)
    y : 라벨 벡터            (N,)
    
    Returns:
    --------
    x[r] : x를 셔플한 np.array (N, ...)
    y[r] : y를 셔플한 np.array (N,)
    """
    
    n = x.shape[0]
    
    r = np.arange(n)
    np.random.shuffle(r)
    
    return x[r], y[r]

def next_batch(x, y, batch_size):
    """
    x, y를 해당 batch_size만큼 잘라서 배치를 생성해주는 generator
    
    Arguments:
    ----------
    x : features 데이터 행렬 (N, ...)
    y : 라벨 벡터            (N,)
    
    Returns:
    --------
    x_batch : x를 배치단위로 자른 것
    y_batch : y를 배치단위로 자른 것.
    """
    
    n = x.shape[0]
    n_batches = int(np.ceil(n / batch_size))
    
    for b in range(n_batches):
        start = b*batch_size
        end = min(n, (b+1)*batch_size)
        
        yield x[start:end], y[start:end]

def load_mnist_data():
    trainset, testset = load_data()
    x_train, y_train = trainset
    x_test, y_test = testset

    x_train = x_train.reshape(*x_train.shape, 1)
    x_test = x_test.reshape(*x_test.shape, 1)

    # mean = 0, std = 0.5
    x_train = (x_train.astype(np.float32) - 128) / 256
    x_test = (x_test.astype(np.float32) - 128) / 256

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    
    return x_train, y_train, x_test, y_test
