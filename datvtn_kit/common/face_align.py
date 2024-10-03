import cv2
from cv2.typing import Size
from numpy.typing import NDArray
from typing import Any, Tuple

import numpy as np
from enum import Enum

class FaceWarp(str, Enum):
    CodeFormer = "facexlib"
    GFPGAN = "facexlib"
    ARCFACE_112_V1 = "arcface_112_v1"
    ARCFACE_112_V2 = "arcface_112_v2"
    ARCFACE_128_V2 = "arcface_128_v2"
    FFHQ_512 = "ffhq_512"


WARP_TEMPLATES = {
	'arcface_112_v1': np.array(
	[
		[ 0.35473214, 0.45658929 ],
		[ 0.64526786, 0.45658929 ],
		[ 0.50000000, 0.61154464 ],
		[ 0.37913393, 0.77687500 ],
		[ 0.62086607, 0.77687500 ]
	]),
	'arcface_112_v2': np.array(
	[
		[ 0.34191607, 0.46157411 ],
		[ 0.65653393, 0.45983393 ],
		[ 0.50022500, 0.64050536 ],
		[ 0.37097589, 0.82469196 ],
		[ 0.63151696, 0.82325089 ]
	]),
	'arcface_128_v2': np.array(
	[
		[ 0.36167656, 0.40387734 ],
		[ 0.63696719, 0.40235469 ],
		[ 0.50019687, 0.56044219 ],
		[ 0.38710391, 0.72160547 ],
		[ 0.61507734, 0.72034453 ]
	]),
	'ffhq_512': np.array(
	[
		[ 0.37691676, 0.46864664 ],
		[ 0.62285697, 0.46912813 ],
		[ 0.50123859, 0.61331904 ],
		[ 0.39308822, 0.72541100 ],
		[ 0.61150205, 0.72490465 ]
	]),
    'facexlib': np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118]
    ])
}


class Aligner:
    """Class for face alignment using different templates and warping methods."""
    
    def estimate_matrix_by_face_landmark(
        self, face_landmark_5: np.ndarray, warp_template: str, crop_size: Size
    ) -> NDArray[Any]:
        """Estimates the affine transformation matrix for face landmarks.

        Args:
            face_landmark_5 (np.ndarray): Array of 5 face landmarks.
            warp_template (str): Template to warp the face to.
            crop_size (Size): Desired output size.

        Returns:
            NDArray[Any]: Affine transformation matrix.
        """
        if warp_template in (FaceWarp.CodeFormer, FaceWarp.GFPGAN):
            normed_warp_template = WARP_TEMPLATES.get(warp_template)
        else:
            normed_warp_template = WARP_TEMPLATES.get(warp_template) * crop_size
        affine_matrix = cv2.estimateAffinePartial2D(
            face_landmark_5, normed_warp_template, method=cv2.RANSAC, ransacReprojThreshold=100
        )[0]
        return affine_matrix

    def warp_face(
        self, temp_vision_frame: np.ndarray, face_landmark_5: np.ndarray, warp_template: str, crop_size: Size
    ) -> Tuple[np.ndarray, NDArray[Any]]:
        """Warps the face based on the provided landmarks and template.

        Args:
            temp_vision_frame (np.ndarray): Input frame/image.
            face_landmark_5 (np.ndarray): Array of 5 face landmarks.
            warp_template (str): Template to warp the face to.
            crop_size (Size): Desired output size.

        Returns:
            Tuple[np.ndarray, NDArray[Any]]: Warped image and the affine transformation matrix.
        """
        affine_matrix = self.estimate_matrix_by_face_landmark(face_landmark_5, warp_template, crop_size)
        crop_vision_frame = cv2.warpAffine(
            temp_vision_frame, affine_matrix, crop_size, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA
        )
        return crop_vision_frame, affine_matrix

    def paste_back(
        self, temp_vision_frame: np.ndarray, crop_vision_frame: np.ndarray, crop_mask: np.ndarray, affine_matrix: NDArray[Any]
    ) -> np.ndarray:
        """Pastes the cropped and warped face back onto the original frame.

        Args:
            temp_vision_frame (np.ndarray): Original frame.
            crop_vision_frame (np.ndarray): Warped/cropped face.
            crop_mask (np.ndarray): Mask used for pasting.
            affine_matrix (NDArray[Any]): Affine matrix used for warping.

        Returns:
            np.ndarray: Final frame with the face pasted back.
        """
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        temp_size = temp_vision_frame.shape[:2][::-1]
        inverse_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_size).clip(0, 1)
        inverse_vision_frame = cv2.warpAffine(crop_vision_frame, inverse_matrix, temp_size, borderMode=cv2.BORDER_REPLICATE)

        paste_vision_frame = temp_vision_frame.copy()
        for i in range(3):  # Loop through each color channel (R, G, B)
            paste_vision_frame[:, :, i] = (
                inverse_mask * inverse_vision_frame[:, :, i] + (1 - inverse_mask) * temp_vision_frame[:, :, i]
            )
        return paste_vision_frame