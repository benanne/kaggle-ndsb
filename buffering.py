import multiprocessing as mp
import Queue
import threading
 
def buffered_gen_mp(source_gen, buffer_size=2):
    """
    Generator that runs a slow source generator in a separate process.
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer size is 2!")
 
    buffer = mp.Queue(maxsize=buffer_size - 1)
    # the effective buffer size is one less, because the generation process
    # will generate one extra element and block until there is room in the buffer.
 
    def _buffered_generation_process(source_gen, buffer):
        for data in source_gen:
            buffer.put(data, block=True)
        buffer.put(None) # sentinel: signal the end of the iterator
        buffer.close() # unfortunately this does not suffice as a signal: if buffer.get()
        # was called and subsequently the buffer is closed, it will block forever.
 
    process = mp.Process(target=_buffered_generation_process, args=(source_gen, buffer))
    process.start()
 
    for data in iter(buffer.get, None):
        yield data


def buffered_gen_threaded(source_gen, buffer_size=2):
    """
    Generator that runs a slow source generator in a separate thread. Beware of the GIL!
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer size is 2!")
 
    buffer = Queue.Queue(maxsize=buffer_size - 1)
    # the effective buffer size is one less, because the generation process
    # will generate one extra element and block until there is room in the buffer.
 
    def _buffered_generation_thread(source_gen, buffer):
        for data in source_gen:
            buffer.put(data, block=True)
        buffer.put(None) # sentinel: signal the end of the iterator
 
    thread = threading.Thread(target=_buffered_generation_thread, args=(source_gen, buffer))
    thread.daemon = True
    thread.start()

    for data in iter(buffer.get, None):
        yield data
