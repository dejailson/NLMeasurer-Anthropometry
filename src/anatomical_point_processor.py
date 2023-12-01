from __future__ import annotations
from src.model.pose_estimation_model import PoseEstimationModel
from src.movenet.processor_movenet import ProcessorMovenet
from src.util.util import Util
from src.util.cvs_converter import CVSConverter
from src.posenet.processor_posenet import ProcessorPosenet


# https://github.com/jeancomp/OpenDPMH
class AnatomicalPointProcessor():
    
    def __init__(self,) :
        self.movenet_processor = ProcessorMovenet()
        
    @staticmethod
    def create()->AnatomicalPointProcessor:
        return AnatomicalPointProcessor()
        
    def process_all_models(self, image_root_dir, dir_output):
        
        for model in PoseEstimationModel._member_map_.values():
            self.process_one_model(image_root_dir=image_root_dir,dir_output=dir_output,model_ps=model)
    
    def process_one_model(self, image_root_dir, dir_output,model_ps:PoseEstimationModel):
        samples = []
        if model_ps == PoseEstimationModel.POSENET:
            samples = self._posenet_process(image_root_dir)
        else:
            samples = self._movenet_process(image_dir=image_root_dir,model_ps=model_ps)
            
        if len(samples) > 0:
            CVSConverter.convert_to_csv(name_model=model_ps.name,dir_output=dir_output,samples=samples)
    
    def _movenet_process(self, image_dir,model_ps:PoseEstimationModel):
        image_paths = Util.list_images(image_dir)
        samples = []
        for image_path in image_paths:
            sample = self.movenet_processor.process(image_path=image_path,model_ps=model_ps)
            samples.append(sample)
        return samples
       

    def _posenet_process(self, image_dir):
        images = Util.list_images(image_dir)
        samples = list(map(lambda image:ProcessorPosenet.process(image), images))
    
        return samples