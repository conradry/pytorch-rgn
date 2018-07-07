"""
Pytorch dataset class and collate function for building the dataloader
"""

import torch
import bcolz
import re, collections
import numpy as np
import utils
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence


def pad_sequence(sequences, batch_first=False):
    r"""
    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *`` where `T` is the
            length of longest sequence.
        Function assumes trailing dimensions and type of all the Tensors
            in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements.

    Returns:
        Tensor of size ``T x B x *`` if batch_first is False
        Tensor of size ``B x T x *`` otherwise
    """
    #sort the sequences from largest to smallest length
    lengths = np.array([sequence.size(0) for sequence in sequences])
    order = np.argsort(lengths)[::-1]
    #sorted_seqs = sorted(sequences, key=lambda x: len(x), reverse=True)

    #use size of largest sequence
    max_size = sequences[order[0]].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
        
    #create an empty tensor of the desired output size
    #use negative one to prevent errors when using embeddings
    #if trailing_dims: 
    out_tensor = torch.zeros(out_dims)
    #else:
    #    out_tensor = 20*torch.ones(out_dims)
    
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

def sequence_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 1, out=out).float()
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
                
            sequences = [torch.from_numpy(b).float() for b in batch]
            padded = pad_sequence(sequences)
            return padded
            
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: sequence_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [sequence_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

class ProteinDataset(Dataset):
    def __init__(self, bcolz_path, encoding, indices=None):
        """encoding has 3 states, None, onehot, and tokens"""
        self.encoding = encoding
        #TODO: in the future, when using multiple different protein chain lengths
        #try using pytorch ConcatDataset class
        if indices is None:
            self.data_array = bcolz.carray(rootdir=bcolz_path)
        else:
            self.data_array = bcolz.carray(rootdir=bcolz_path)[indices]
    
    def __len__(self):
        return len(self.data_array)
    
    def __getitem__(self, idx):
        
        name, sequence, coords = self.data_array[idx]
        length = len(sequence[0])
        sequence_vec = sequence
        
        if self.encoding == 'onehot':
            sequence_vec = utils.encode_sequence(sequence, onehot=True)
        elif self.encoding == 'tokens':
            sequence_vec = utils.encode_sequence(sequence, onehot=False)
            
        sample = {'name': name,
                  'sequence': sequence_vec,
                  'coords': coords,
                  'length': length}
                                    
        return sample
    
class ProteinNet(Dataset):
    def __init__(self, bcolz_path):
        self.data_array = bcolz.carray(rootdir=bcolz_path)
    
    def __len__(self):
        return len(self.data_array)
    
    def __getitem__(self, idx):
        
        name, sequence, pssm, coords, mask = self.data_array[idx]
        length = len(sequence)
        sequence_vec = utils.encode_sequence(sequence, onehot=True)
        seq_pssm = np.concatenate([sequence_vec, pssm], axis=1)
            
        sample = {'name': name,
                  'sequence': seq_pssm,
                  'coords': coords,
                  'length': length,
                  'mask': mask
                 }
                                    
        return sample
    