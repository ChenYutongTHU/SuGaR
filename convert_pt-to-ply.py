import torch, sys, os
from plyfile import PlyData, PlyElement
import numpy as np

def construct_list_of_attributes(f_dc, f_rest, scale, rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    for i in range(f_rest.shape[1]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scale.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l

ckpt_file = sys.argv[1]
# ply_file = os.path.join(os.path.dirname(ckpt_file),
#                          os.path.basename(ckpt_file).split('.')[0] + '.ply')
ply_file = sys.argv[2]
os.makedirs(os.path.dirname(ply_file), exist_ok=True)
state_dict = torch.load(ckpt_file, map_location='cpu')['state_dict']
# ['_points', 'all_densities', '_scales', '_quaternions',
#     '_sh_coordinates_dc', '_sh_coordinates_rest']
xyz = state_dict['_points'].numpy()
normals = np.zeros_like(xyz)
f_dc = state_dict['_sh_coordinates_dc'].transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
f_rest = state_dict['_sh_coordinates_rest'].transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
opacities = state_dict['all_densities'].cpu().numpy()
scale = state_dict['_scales'].cpu().numpy()
rotation = state_dict['_quaternions'].cpu().numpy()

dtype_full = [(attribute, 'f4')
              for attribute in construct_list_of_attributes(
                    f_dc, f_rest, scale, rotation
              )]

elements = np.empty(xyz.shape[0], dtype=dtype_full)
attributes = np.concatenate(
    (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
elements[:] = list(map(tuple, attributes))
el = PlyElement.describe(elements, 'vertex')
PlyData([el]).write(ply_file)
print(ply_file)

