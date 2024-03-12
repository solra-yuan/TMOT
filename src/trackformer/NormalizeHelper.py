import torch

class NormalizeHelper:
    @staticmethod
    def per_channel(image):
        """Normalize each channel of the image tensor independently."""
        # image shape: [C, H, W]
        min_val = image.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_val = image.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        
        # Normalize each channel independently
        normalized_image = (image - min_val) / (max_val - min_val)
        
        return normalized_image

    @staticmethod
    def across_channels(image):
        """Normalize the image tensor across all channels."""
        # image shape: [C, H, W]
        min_val = image.min()
        max_val = image.max()
        
        # Normalize across all channels
        normalized_image = (image - min_val) / (max_val - min_val)
        
        return normalized_image