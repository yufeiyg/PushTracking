for seed in "D_shape_texture" "O_shape_texture"; do
    python run_custom.py --use_segmenter 1 --use_gui 1 --debug_level 2 --object_name $seed
done