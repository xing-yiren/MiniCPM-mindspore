import unittest
import numpy as np
import transformers
import torch
import mindspore
from mindspore import nn, ops

from model.configuration_minicpm import MiniCPMConfig as MSConfig
from model_torch.configuration_minicpm_torch import MiniCPMConfig as TorchConfig

  
class MiniCPMTest(unittest.TestCase): 
    def get_input(self, batch_size, seq_len, hidden_size):
        input_array = np.random.rand(batch_size, seq_len, hidden_size)
        input_array = input_array.astype(np.float32)
        ms_input = mindspore.Tensor.from_numpy(input_array)
        torch_input = torch.from_numpy(input_array)
        return ms_input, torch_input

    def test_rms_norm(self):
        from model.modelling_minicpm import MiniCPMRMSNorm as MSRMSNorm
        from model_torch.modelling_minicpm_torch import MiniCPMRMSNorm as TorchRMSNorm
        ms_input, torch_input = self.get_input(1, 10, 4096)
        ms_rms_norm = MSRMSNorm()
        torch_rms_norm = TorchRMSNorm()
        ms_output = ms_rms_norm(ms_input)
        torch_output = torch_rms_norm(torch_input)
        is_close = np.allclose(ms_output.asnumpy(), torch_output.detach().numpy(), atol=1e-3)
        self.assertTrue(is_close, msg="MindSpore和PyTorch中NMSNorm的输出在指定的容差范围内不相等")

  
if __name__ == '__main__':  
    unittest.main()