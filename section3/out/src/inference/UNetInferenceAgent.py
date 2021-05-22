"""
Contains class that runs inferencing
"""
import torch
import torch.nn.functional as F
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        
        volume_padded = med_reshape(volume, (volume.shape[0], self.patch_size, self.patch_size))
        return self.single_volume_inference(volume_padded)[:volume.shape[0],:volume.shape[1],:volume.shape[2]]

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        vol_tensor = torch.from_numpy(volume).type(torch.FloatTensor).to(self.device)
        vol_tensor = vol_tensor.unsqueeze(1)
        # v now has shape (num_sagittal_slices,1,patch_size,patch_size)
        # 0th index is being used as batch index. so the entire volume is being treated as
        # one big batch of slices
        with torch.no_grad():
            prediction = F.softmax(self.model(vol_tensor), dim=1).cpu().numpy()
        # prediction has shape (num_sagittal_slices,3,patch_size,patch_size)

        return prediction.argmax(axis=1)
