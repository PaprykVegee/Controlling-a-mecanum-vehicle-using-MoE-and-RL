import cv2
import numpy as np
import ast


class ROSMaskGenerator:
    def __init__(self, tolerance: int, target_color: tuple, rgb_dict: dict):
        self.tolerance = tolerance
        self.target_color = target_color
        self.rgb_dict = rgb_dict

    def __color_mask(self, img: np.ndarray, color: tuple[int, int, int], tol: int) -> np.ndarray:
        """Zwraca maskę uint8 (0/255) dla pikseli w tolerancji RGB"""
        color = np.array(color, dtype=np.int16)
        img_int = img.astype(np.int16)

        diff = np.abs(img_int - color)
        mask = np.all(diff <= tol, axis=-1)

        return (mask.astype(np.uint8)) * 255  

    def process(self, img: np.ndarray, mode: str = "BGR") -> np.ndarray:
        if mode == "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        road_mask       = self.__color_mask(img, self.target_color[0], self.tolerance)
        roadline_mask   = self.__color_mask(img, self.target_color[1], self.tolerance)
        wall_mask       = self.__color_mask(img, self.target_color[2], self.tolerance)
        vege_mask       = self.__color_mask(img, self.target_color[3], self.tolerance + 5)
        sidewalk_mask   = self.__color_mask(img, self.target_color[4], self.tolerance + 15)

        kernel = np.ones((5,5), np.uint8)
        wall_mask     = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
        vege_mask     = cv2.morphologyEx(vege_mask, cv2.MORPH_CLOSE, kernel)
        sidewalk_mask = cv2.morphologyEx(sidewalk_mask, cv2.MORPH_CLOSE, kernel)

        mask = np.zeros_like(img, dtype=np.uint8)

        for msk, key in zip(
            [vege_mask, wall_mask, sidewalk_mask, road_mask, roadline_mask],
            ['Vegetation', 'Wall', 'Sidewalk', 'Road', 'Road Line']
        ):
            apply_mask = msk > 0
            apply_mask &= np.all(mask == 0, axis=-1)
            mask[apply_mask] = self.rgb_dict[key]

        return mask

    

def perspectiveWarp(frame):

    height, width = frame.shape[:2]
    y_sc = 0.6 
    x_sc = 0.3399 
    H2 = int(height * y_sc)
    W2_L = int(width * x_sc)
    W2_R = int(width * (1 - x_sc))
    
    src = np.float32([
        [W2_L, H2], 
        [W2_R, H2], 
        [width, height],
        [0, height]
    ])

    dst = np.float32([
        [0, 0],             
        [width, 0],         
        [width, height],    
        [0, height]         
    ])
    img_size = (width, height)
    matrix = cv2.getPerspectiveTransform(src, dst)
    birdseye = cv2.warpPerspective(frame, matrix, img_size)
    return birdseye