import cv2
from albumentations.core.transforms_interface import DualTransform

class LetterBox(DualTransform):
    def __init__(self, target_size=(64, 64), fill_color=(255, 255, 255), p=1.0):
        super().__init__(p)
        self.target_size = target_size
        self.fill_color = fill_color
    
    def apply(self, img, **params):
        return self._letterbox(img, self.fill_color, is_mask=False)

    def apply_to_mask(self, img, **params):
        return self._letterbox(img, 0, is_mask=True)

    def _letterbox(self, img, pad_val, is_mask):
        # Handle both HWC and HW formats
        if len(img.shape) == 3:
            img_h, img_w = img.shape[:2]
        else:
            img_h, img_w = img.shape

        target_h, target_w = self.target_size

        scale = min(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded_img = cv2.copyMakeBorder(
            resized_img, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=pad_val
        )
        return padded_img

    def get_transform_init_args_names(self):
        return ("target_size", "fill_color")
