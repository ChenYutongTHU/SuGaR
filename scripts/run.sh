# cd gaussian_splatting/submodules/diff-gaussian-rasterization/
# pip install -e .
# cd ../simple-knn/
# pip install -e .
# cd ../../../

for inverse_ratio in 1
do
ratio=$(echo "scale=6; 1/$inverse_ratio" | bc)

for split in train_single-scale
do
python gaussian_splatting/train.py \
    -s ~/3D_projects/data/BlenderNeRF/1st_scene_v2_rgba \
    --rnd_background \
    --blender_bbox 5 5 5 -5 -5 -1 \
    --blender_train_json ${split}.json \
    --blender_test_jsons 'fly_focal100.json'  \
    --sh_degree 0 \
    --train_num_camera_ratio ${ratio} \
    -m output_from-gs/1st_scene_v2_rgba/1-${inverse_ratio}/${split}-1-${inverse_ratio}_sh-0 \
    --dataset_type loader --eval \
    --iterations 7000
done
done