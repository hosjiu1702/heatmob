from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from torch import nn
import numpy as np
from PIL import Image


class Masker:
    def __init__(self, model_id='mattmdjaga/segformer_b2_clothes'): # SegFormer
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id)

    def get_binary_mask(self, image, return_pil=False):
        inputs = self.processor(images=image, return_tensors="pt")        
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0] # Dunno how this line work
        np_image = np.array(image)
        np_image[pred_seg != 4] = 0
        np_image[pred_seg == 4] = 255
        binary_mask = ((pred_seg == 4) * 255).numpy().astype(np.uint8)

        return Image.fromarray(binary_mask) if return_pil else binary_mask