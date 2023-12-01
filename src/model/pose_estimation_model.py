from enum import Enum
from typing import List

class PoseEstimationModel(Enum):
    POSENET = 1
    MOVENET_LIGHTNING_V4_MULTIPOSE = 2
    MOVENET_LIGHTNING_INT_8 = 3
    MOVENET_LIGHTNING_FLOAT_16 = 4
    MOVENET_THUNDER_V4_SINGLEPOSE = 5
    MOVENET_THUNDER_INT_8 = 6
    MOVENET_THUNDER_FLOAT_16 = 7
    MOVENET_LIGHTNING_V4_8 = 8
    MOVENET_LIGHTNING_V4_16 = 9
    MOVENET_THUNDER_V4_8 = 10
    MOVENET_THUNDER_V4_16 = 11

    @classmethod
    def movenet_models(cls)->List['PoseEstimationModel']:
        return list(filter(lambda model: model != PoseEstimationModel.POSENET,PoseEstimationModel._member_map_.values()))
    
    @property
    def movenet_model(self):
        if self == self.POSENET:
            return None
        elif self == self.MOVENET_LIGHTNING_V4_MULTIPOSE:
            return 'https://tfhub.dev/google/movenet/multipose/lightning/1'
        elif self == self.MOVENET_LIGHTNING_INT_8:
            return 'ml_models/lite-model_movenet_singlepose_lightning_tflite_int8.tflite'
        elif self == self.MOVENET_LIGHTNING_V4_8:
            return 'ml_models/lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite'
        elif self == self.MOVENET_LIGHTNING_V4_16:
            return 'ml_models/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite'
        elif self == self.MOVENET_LIGHTNING_FLOAT_16:
            return 'ml_models/movenet_singlepose_lightning_float16.tflite'
        elif self == self.MOVENET_THUNDER_V4_SINGLEPOSE:
            return 'https://tfhub.dev/google/movenet/singlepose/thunder/4'
        elif self == self.MOVENET_THUNDER_INT_8:
            return 'ml_models/lite-model_movenet_singlepose_thunder_tflite_int8.tflite'
        elif self == self.MOVENET_THUNDER_FLOAT_16:
            return 'ml_models/movenet_singlepose_thunder_float16.tflite'
        elif self == self.MOVENET_THUNDER_V4_8:
            return 'ml_models/lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite'
        elif self == self.MOVENET_THUNDER_V4_16:
            return 'ml_models/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite'
        