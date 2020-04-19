import time

def timeit(text):
    def decorator(f):
        def wrap(*args, **kwargs):
            print('{} is running...'.format(text))
            start = time.time()
            func = f(*args, **kwargs)
            end = time.time()
            print('{:s} function took {:.3f} s'.format(f.__name__, (end-start)))
            return func
        return wrap
    return decorator



