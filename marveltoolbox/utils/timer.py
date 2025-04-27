import time

class Timer:
    def __init__(self, steps=-1, logger=None):
        self.steps = steps
        self.step_now = 0
        self.time_begin = time.time()
        self.time_last_step = self.time_begin
        self.time_now = self.time_begin
        self.step_begin_time = self.time_begin
        self.step_end_time = self.time_begin
        self.step_time_all = 0.0
        self.logger = logger
    
    def step(self, interval=1):
        self.step_now += interval
        remain_time, per_step_time = self._remain_time()
        h, m, s = self._convert(remain_time)
        print_str = 'Epoch: {:0>3d}/{:0>3d} {:.0f}% ' \
                    '[Remain:{:2d}h/{:2d}m/{:2d}s | Avg: {:.2f}s/Epoch]\n'.format(
                        self.step_now, 
                        self.steps, 
                        self.step_now*100/self.steps,
                        h,m,s,
                        per_step_time,)
        print(print_str)
        if self.logger is not None:
            self.logger.info(print_str)
                    

    def init(self):
        self.time_begin = time.time()
        self.time_last_step = self.time_begin
        self.time_now = self.time_begin
        self.step_begin_time = self.time_begin
        self.step_end_time = self.time_begin

    def step_begin(self):
        self.step_begin_time = time.time()

    def step_end(self):
        self.step_end_time = time.time()
        self.step_time_all += (self.step_end_time - self.step_begin_time)
            
    def _remain_time(self):
        self.time_now = time.time() 
        time_consume = (self.time_now-self.time_begin)
        remain_steps = self.steps - self.step_now
        per_step_time = time_consume/self.step_now
        remain_time = remain_steps * per_step_time
        per_step_time = self.step_time_all / self.step_now
        return remain_time, per_step_time
        
    def _convert(self, diff_time):
        m, s = divmod(int(diff_time), 60)
        h, m = divmod(m, 60)
        return h, m, s
    
    def begin(self):
        self.time_begin = time.time()
        
    def end(self, is_print=True):
        diff_time = time.time() - self.time_begin
        h, m, s = self._convert(diff_time)
        self.diff_time = diff_time
        self.record = 'Consume:[{:2d}h/{:2d}m/{:2d}s]\n'.format(h, m, s)
        if is_print:
            print(self.record)
        
    def __str__(self):
        h, m, s = self._convert(self.diff_time)
        return 'Consume{:2d}h{:2d}m{:2d}s'.format(h, m, s)