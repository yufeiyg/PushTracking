for seed in "egg_carton" "green_tea" "expo_box" "milk"; do
    python run_custom.py --use_segmenter 1 --use_gui 1 --debug_level 2 --object_name $seed
done