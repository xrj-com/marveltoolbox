import torch

class BaseConfs:
    def __init__(self):
        self.chkpt_path = './chkpts'
        self.log_path = './logs'
        self.batch_size = 128
        self.epochs = 50
        self.seed = 0

        self.get_dataset()
        self.get_flag()
        self.get_device()
        print(self)

    def get_dataset(self):
        self.dataset = 'mnist'

    def get_flag(self):
        self.flag = 'demo-{}'.format(self.dataset)

    def get_device(self):
        self.device_ids = [0]
        self.ngpu = len(self.device_ids)
        self.device = torch.device(
            "cuda:{}".format(self.device_ids[0]) if \
            (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
                    

    def __str__(self):
        print_str = 'Configs：\n'
        print_str += 'Flag:       {}\n'.format(self.flag)
        print_str += 'Batch size: {}\n'.format(self.batch_size)
        print_str += 'Epochs:     {}\n'.format(self.epochs)
        print_str += 'device:     {}\n'.format(self.device)
        return print_str


class BaseExpConfs:
    def __init__(self):
        self.exp_flag = 'Exp_demo'
        self.exp_path = './exps'
        self.batch_size = 128
        self.seed = 0

    def __str__(self):
        print_str = 'Configs: \n'
        print_str += 'Exp flag:   {}\n'.format(self.exp_flag)
        print_str += 'Exp path:   {}\n'.format(self.exp_path)
        print_str += 'Seed:       {}\n'.format(self.seed)
        return print_str


if __name__ == '__main__':
    args = Args()