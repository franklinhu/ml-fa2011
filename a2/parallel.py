import multiprocessing as mp
import Queue

class Workers(object):

    def __init__(self):
        self.output_queue = mp.JoinableQueue()
        self.input_queue = mp.JoinableQueue()
        self.fn = None
        self.started = False
        self.workers = []

    def initialize_n_workers(self, n):
        for _ in range(n): 
            w = Worker(input_queue=self.input_queue, output_queue=self.output_queue)
            w.daemon = True
            self.workers.append(w)

    def set_function(self, fn):
        self.fn = fn
        for worker in self.workers:
            worker.fn = self.fn

    def start(self):
        if not self.started:
            self.started = True
            for worker in self.workers:
                worker.start()

    def terminate(self):
        for worker in self.workers:
            worker.terminate()

    def _add_inputs(self, inputs):
        for job in inputs:
            self.input_queue.put(job)

    def run_over_data(self, inputs=None):
        self._add_inputs(inputs)
        self.input_queue.join()
        outputs = []
        while len(outputs) < len(inputs):
            output = self.output_queue.get()
            outputs.append(output)
        
        return outputs

class Worker(mp.Process):
    
    def __init__(self, fn = None, input_queue=None, output_queue=None):
        mp.Process.__init__(self)
        self.fn = fn
        self.input_queue = input_queue
        self.output_queue = output_queue
    
    def run(self):
        while True:
            try:
                args = self.input_queue.get()
            except Queue.Empty:
                break
            self.output_queue.put(self.fn(*args))
            self.input_queue.task_done()
