import torch
import pytorch_tools as pt
from torchvision.utils import make_grid


class PredictViewer(pt.fit_wrapper.callbacks.TensorBoard):
    """Saves first batch and visualizes model predictions on it for every epoch"""

    def __init__(self, log_dir, log_every=50, num_images=4):
        """num_images (int): number of images to visualize"""
        super().__init__(log_dir, log_every=20)
        self.has_saved = False  # Flag to save first batch
        self.img_batch = None
        self.num_images = num_images

    def on_batch_end(self):
        super().on_batch_end()
        # save first val batch
        if not self.has_saved and not self.state.is_train:
            # Take `num_images` from batch
            self.img_batch = self.state.input[0].detach()[:self.num_images] 
            self.has_saved = True
            target_batch = self.state.input[1].detach()[:self.num_images, :1]
            self.target_grid = make_grid((target_batch * 255).type(torch.uint8), nrow=self.num_images)

    def on_epoch_end(self):
        super().on_epoch_end()
        self.state.model.eval()  # not sure if needed but just in case
        pred = self.state.model(self.img_batch)
        pred = (pred.sigmoid() * 255).type(torch.uint8)
        grid = make_grid(pred, nrow=self.num_images)
        grid = torch.cat([grid, self.target_grid], axis=1)
        self.writer.add_image("val/prediction", grid, self.current_step)