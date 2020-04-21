import torch
import torch.optim as optim
from tqdm.auto import tqdm

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_

from moses.interfaces import MosesTrainer
from moses.utils import OneHotVocab, Logger, CircularBuffer, set_torch_seed_to_all_gens
from moses.vae.misc import CosineAnnealingLRWithRestart, KLAnnealer


class BindingTrainer(MosesTrainer):
    def __init__(self, config):
        self.config = config
    
    def get_vocabulary(self, data):
        return OneHotVocab.from_data(data)
    
    def get_collate_fn(self, model):
        device = self.get_collate_device(model)
        
        def collate(data):
            # data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device)
                       for string in data[1]]
            
            return tensors
        
        return collate
    
    def get_dataloader(self, model, data, collate_fn=None, shuffle=True):
        collate_fn = self.get_collate_fn(model)
        return DataLoader(TensorDataset(*data), batch_size=self.config.n_batch,
                          shuffle=shuffle,
                          num_workers=self.n_workers, collate_fn=collate_fn,
                          worker_init_fn=set_torch_seed_to_all_gens
                          if self.n_workers > 0 else None)
    
    def _train_epoch(self, model, epoch, tqdm_data, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()
        
        loss_values = CircularBuffer(self.config.n_last)
        for input_batch in tqdm_data:
            print(input_batch)
            input_batch = tuple(data.to(model.device) for data in input_batch)
            
            # Forward
            loss = model(input_batch)
            
            # Backward
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.get_optim_params(model),
                                self.config.clip_grad)
                optimizer.step()
            
            # Log
            loss_values.add(loss.item())
            lr = (optimizer.param_groups[0]['lr']
                  if optimizer is not None
                  else 0)
            
            # Update tqdm
            loss_value = loss_values.mean()
            postfix = [f'loss={loss_value:.5f}',
                       f'lr={lr:.5f}']
            tqdm_data.set_postfix_str(' '.join(postfix))
        
        postfix = {
            'epoch': epoch,
            'lr': lr,
            'loss': loss_value,
            'mode': 'Eval' if optimizer is None else 'Train'}
        
        return postfix
    
    def get_optim_params(self, model):
        return (p for p in model.vae.parameters() if p.requires_grad)
    
    def _train(self, model, train_loader, val_loader=None, logger=None):
        device = model.device
        n_epoch = self._n_epoch()
        
        optimizer = optim.Adam(self.get_optim_params(model),
                               lr=self.config.lr_start)
        lr_annealer = CosineAnnealingLRWithRestart(optimizer,
                                                   self.config)
        
        model.zero_grad()
        for epoch in range(n_epoch):
            # Epoch start
            tqdm_data = tqdm(train_loader,
                             desc='Training (epoch #{})'.format(epoch))
            postfix = self._train_epoch(model, epoch, tqdm_data, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)
            
            if val_loader is not None:
                tqdm_data = tqdm(val_loader,
                                 desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, epoch, tqdm_data)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)
            
            if (self.config.model_save is not None) and \
                    (epoch % self.config.save_frequency == 0):
                model = model.to('cpu')
                torch.save(model.state_dict(),
                           self.config.model_save[:-3] +
                           '_{0:03d}.pt'.format(epoch))
                model = model.to(device)
            
            # Epoch end
            lr_annealer.step()
    
    def fit(self, model, train_data, val_data=None):
        logger = Logger() if self.config.log_file is not None else None
        
        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(
            model, val_data, shuffle=False
        )
        
        self._train(model, train_loader, val_loader, logger)
        return model
    
    def _n_epoch(self):
        return sum(
            self.config.lr_n_period * (self.config.lr_n_mult ** i)
            for i in range(self.config.lr_n_restarts)
        )
