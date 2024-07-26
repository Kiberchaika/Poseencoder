
from super_gradients.training import models
from super_gradients.common.object_names import Models
from tqdm import tqdm
from typing import Iterable, Optional, Union
from super_gradients.training.pipelines.pipelines import PoseEstimationPipeline 
from super_gradients.training.utils.predict import (
    ImagePoseEstimationPrediction,
    ImagesPoseEstimationPrediction,
    PoseEstimationPrediction,
)
import sys

def _combine_image_prediction_to_images(
    self, images_predictions: Iterable[PoseEstimationPrediction], n_images: Optional[int] = None
) -> Union[ImagesPoseEstimationPrediction, ImagePoseEstimationPrediction]:
    images_predictions = [image_predictions for image_predictions in tqdm(images_predictions, total=n_images, desc="Predicting Images", disable=True)]
    images_predictions = ImagesPoseEstimationPrediction(_images_prediction_lst=images_predictions)

    return images_predictions

# Replace the method in the existing PoseEstimationPipeline class
PoseEstimationPipeline._combine_image_prediction_to_images = _combine_image_prediction_to_images

# Detector class to wrap around YOLO detector
class YOLODetector: 
    def __init__(self):
        self.model = models.get(Models.YOLO_NAS_POSE_M, pretrained_weights="coco_pose").cuda()
        sys.stdout = sys.__stdout__

    def predict(self, frames, conf=0.6, iou=0.7):
        model_predictions = self.model.predict(frames, conf=conf, iou=iou, pre_nms_max_predictions=300, post_nms_max_predictions=20, fuse_model=False)
        poses = [prediction.prediction.poses for prediction in model_predictions]
        return poses
    




