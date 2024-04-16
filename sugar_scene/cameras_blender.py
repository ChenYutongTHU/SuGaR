# from gaussian_splatting.scene.dataset_readers import readCamerasFromTransforms
# from gaussian_splatting.utils.camera_utils import loadCam, cameraList_from_camInfos

import sys
sys.path.append('../')
from mip_splatting.scene.dataset_readers import readCamerasFromTransforms
from mip_splatting.utils.camera_utils import loadCam, cameraList_from_camInfos
class tmp_args:
    def __init__(self, image_resolution, rnd_background, dataset_type):
        self.data_device = 'cuda'
        self.resolution = image_resolution
        self.dataset_type = dataset_type
        self.rnd_background = rnd_background


def load_gs_cameras_blender(
    source_path, 
    blender_json, 
    num_camera_ratio,
    rnd_background, 
    white_background=False, extension='png', 
    dataset_type='list',
    image_resolution=-1, load_gt_images=True, max_img_size=1920, **kwargs):

    assert dataset_type=='list' and image_resolution==-1
    assert rnd_background==False
    json_file = blender_json if blender_json is not None else "transforms_train.json"
    print(f"Reading Transforms from {json_file} ", end=' ')
    cam_infos = readCamerasFromTransforms(
        source_path, json_file, white_background, extension, num_camera_ratio, dataset_type)
    print(
        f'#={len(cam_infos)}(num_camera_ratio={num_camera_ratio})')
    
    shuffle = False
    cam_list = cameraList_from_camInfos(
        cam_infos=cam_infos, resolution_scale=1,  
        args=tmp_args(image_resolution=image_resolution, rnd_background=rnd_background, dataset_type=dataset_type), 
        shuffle=shuffle) #Shuffle???
    
    return cam_list
    '''
    test_json_files = blender_test_jsons.split(
        ',') if blender_test_jsons is not None else ["transforms_test.json"]
    print(f"Reading Test Transforms from {test_json_files}", end=' ')
    test_cam_infos = {}
    if test_json_files != [""]:
        for test_json_file in test_json_files:
            tag = test_json_file.replace('.json', '')
            test_cam_infos[tag] = readCamerasFromTransforms(
                source_path, test_json_file, white_background, extension, dataset_type=dataset_type)
            print(f'{test_json_file}, #={len(test_cam_infos[tag])}')

    if not eval:
        if test_cam_infos == {}:
            test_cam_infos = []
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    '''
