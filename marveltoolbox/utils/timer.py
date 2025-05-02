import time

class Timer:
    def __init__(self, steps=-1, logger=None):
        self.steps = steps
        self.step_now = 0
        self.time_begin = time.time()
        self.time_last_step = self.time_begin
        self.time_now = self.time_begin
        self.eval_begin_dict = {}
        self.eval_end_dict = {}
        self.eval_total_dict = {}
        self.logger = logger
    
    def step(self, interval=1):
        self.step_now += interval
        remain_time, per_step_time = self._remain_time()
        h, m, s = self._convert(remain_time)
        eval_str = '{'
        if self.eval_total_dict:
            for key, value in self.eval_total_dict.items():
                eval_str +=  ' {}: {:.2f}s |'.format(key, value/self.step_now) 
            eval_str = eval_str[:-1] + '}'
        print_str = 'Epoch: {:0>3d}/{:0>3d} {:.0f}% ' \
                    '[Remain:{:2d}h/{:2d}m/{:2d}s | Avg: {:.2f}s{}/Epoch]\n'.format(
                        self.step_now, 
                        self.steps, 
                        self.step_now*100/self.steps,
                        h,m,s,
                        per_step_time, eval_str)
        print(print_str)
        if self.logger is not None:
            self.logger.info(print_str)
                    

    def init(self):
        self.time_begin = time.time()
        self.time_last_step = self.time_begin
        self.time_now = self.time_begin
        self.eval_begin_dict = {}
        self.eval_end_dict = {}
        self.eval_total_dict = {}

    def eval_begin(self, key):
        self.eval_begin_dict[key] = time.time()
        if not key in self.eval_total_dict.keys():
            self.eval_total_dict[key] = 0.0

    def eval_end(self, key):
        self.eval_end_dict[key] = time.time()
        self.eval_total_dict[key] += (self.eval_end_dict[key] - self.eval_begin_dict[key])
            
    def _remain_time(self):
        self.time_now = time.time() 
        time_consume = (self.time_now-self.time_begin)
        remain_steps = self.steps - self.step_now
        per_step_time = time_consume/self.step_now
        remain_time = remain_steps * per_step_time
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