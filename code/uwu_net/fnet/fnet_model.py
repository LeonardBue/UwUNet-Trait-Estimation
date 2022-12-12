import os
import torch
import importlib
import pdb
import time

class Model(object):
    def __init__(
            self,
            nn_module = None,
            init_weights = True,
            lr = 0.001,
            criterion_fn = torch.nn.MSELoss, 
            nn_kwargs={},
            gpu_ids = -1,
            betas = (0.8, 0.91),
            use_amp = True,
    ):
        self.nn_module = nn_module
        self.nn_kwargs = nn_kwargs
        self.init_weights = init_weights
        self.lr = lr
        self.criterion_fn = criterion_fn
        self.count_iter = 0
        self.gpu_ids = [gpu_ids] if isinstance(gpu_ids, int) else gpu_ids
        self.device = torch.device('cuda', self.gpu_ids[0]) if self.gpu_ids[0] >= 0 else torch.device('cpu')
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.dtype = torch.float16# if self.device.type == 'cuda' else torch.bfloat16
        
        self.betas = betas
        self.criterion = criterion_fn()
        self._init_model(nn_kwargs=self.nn_kwargs)

    def _init_model(self, nn_kwargs={}):
        if self.nn_module is None:
            self.net = None
            return
        self.net = importlib.import_module('fnet.nn_modules.' + self.nn_module).Net(**nn_kwargs)
        if self.init_weights:
            self.net.apply(_weights_init)
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=self.betas) 
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9) 
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0.0001)
        
    def __str__(self):
        out_str = '{:s} | {:s} | iter: {:d}'.format(
            self.nn_module,
            str(self.nn_kwargs),
            self.count_iter,
        )
        return out_str

    def get_state(self):
        return dict(
            nn_module = self.nn_module,
            nn_kwargs = self.nn_kwargs,
            nn_state = self.net.state_dict(),
            optimizer_state = self.optimizer.state_dict(),
            count_iter = self.count_iter,
        )

    def to_gpu(self, gpu_ids):
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda', self.gpu_ids[0]) if self.gpu_ids[0] >= 0 else torch.device('cpu')
        self.net.to(self.device)
        _set_gpu_recursive(self.optimizer.state, self.gpu_ids[0])  # this may not work in the future

    def save_state(self, path_save):
        curr_gpu_ids = self.gpu_ids
        dirname = os.path.dirname(path_save)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.to_gpu(-1)
        try:
            torch.save(self.get_state(), path_save)
        except:
            time.sleep(2)
            try:
                torch.save(self.get_state(), path_save)
            except:
                pass
        self.to_gpu(curr_gpu_ids)

    def load_state(self, path_load, gpu_ids=-1):
        state_dict = torch.load(path_load)
        self.nn_module = state_dict['nn_module']
        self.nn_kwargs = state_dict.get('nn_kwargs', {})
        self._init_model(nn_kwargs=self.nn_kwargs)
        self.net.load_state_dict(state_dict['nn_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.count_iter = state_dict['count_iter']
        self.to_gpu(gpu_ids)
    
    def load_spatial(self, path_load, gpu_ids=-1):
        pretrained_dict = torch.load(path_load)
        model_dict = self.get_state()

        # 1. filter out unwanted keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        filtered_state = pretrained_dict['nn_state'].copy()
        for k in pretrained_dict['nn_state'].keys():
            if k.startswith((   'spec_conv',
                                'spec_down_conv', 
                                'spec_down_down_conv',
                                'spec_convt',
                                'spec_bn',
                                'spec_final',
                                'spec_final_pool')):

                filtered_state.pop(k)
        pretrained_dict['nn_state'] = filtered_state
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.nn_module = model_dict['nn_module']
        self._init_model(nn_kwargs=self.nn_kwargs)
        self.net.load_state_dict(model_dict['nn_state'], strict=False)
        self.to_gpu(gpu_ids)

    def do_train_iter(self, signal, target):
        signal = signal.clone().detach().requires_grad_(True).to(self.device)
        target = target.clone().detach().requires_grad_(True).to(self.device)
        if len(self.gpu_ids) > 1:
            module = torch.nn.DataParallel(
                self.net,
                device_ids = self.gpu_ids,
            )
        else:
            module = self.net
        module.train()
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
            output = module(signal)
            if self.net.final_chan == 1:
                loss = self.criterion(output, target.squeeze())
            else:
                loss = self.criterion(output, target)
        if self.device.type == 'cuda':
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.count_iter += 1
        return loss.item()  
    
    def predict(self, signal):
        signal = signal.clone().detach().to(self.device)
        if len(self.gpu_ids) > 1:
            module = torch.nn.DataParallel(
                self.net,
                device_ids = self.gpu_ids,
            )
        else:
            module = self.net
        module.eval()
        with torch.no_grad():
            prediction = module(signal).cpu()
        return prediction

def _weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0) 

def _set_gpu_recursive(var, gpu_id):
    """Moves Tensors nested in dict var to gpu_id.

    Modified from pytorch_integrated_cell.

    Parameters:
    var - (dict) keys are either Tensors or dicts
    gpu_id - (int) GPU onto which to move the Tensors
    """
    for key in var:
        if isinstance(var[key], dict):
            _set_gpu_recursive(var[key], gpu_id)
        elif torch.is_tensor(var[key]):
            if gpu_id == -1:
                var[key] = var[key].cpu()
            else:
                var[key] = var[key].cuda(gpu_id)
