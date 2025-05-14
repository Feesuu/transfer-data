
#!/bin/bash
config_path="./config/config_muq.yaml"

dataset_name=$(yq -r '.dataset_name' $config_path)
file="./data/${dataset_name}/subfolder.txt"
my_gpu=7

subfolder_list=()
while IFS= read -r line; do
  subfolder_list+=("$line")
done < "$file"

for i in "${!subfolder_list[@]}"; do
  if [ -n "${subfolder_list[$i]}" ]; then
    # echo "Processing the sub-folder $i..."
    # source /home/yaodong/miniconda3/bin/activate /home/yaodong/miniconda3/envs/gnnrag
    # CUDA_VISIBLE_DEVICES="$my_gpu" python main.py --config_path "$config_path" --folder_name "${subfolder_list[$i]}" --step 0
    source /home/yaodong/miniconda3/bin/activate /home/yaodong/miniconda3/envs/copy
    CUDA_VISIBLE_DEVICES="$my_gpu" python main.py --config_path "$config_path" --folder_name "${subfolder_list[$i]}" --step 1
    source /home/yaodong/miniconda3/bin/activate /home/yaodong/miniconda3/envs/gnnrag
    CUDA_VISIBLE_DEVICES="$my_gpu" python main.py --config_path "$config_path" --folder_name "${subfolder_list[$i]}" --step 2
    CUDA_VISIBLE_DEVICES="$my_gpu" python main.py --config_path "$config_path" --folder_name "${subfolder_list[$i]}" --step 3
    CUDA_VISIBLE_DEVICES="$my_gpu" python main.py --config_path "$config_path" --folder_name "${subfolder_list[$i]}" --step 4
    #CUDA_VISIBLE_DEVICES="$my_gpu" python main.py --config_path "$config_path" --folder_name "${subfolder_list[$i]}" --step 5
    # CUDA_VISIBLE_DEVICES="$my_gpu" python main.py --config_path "$config_path" --folder_name "${subfolder_list[$i]}" --step 6
    # CUDA_VISIBLE_DEVICES="$my_gpu" python main.py --config_path "$config_path" --folder_name "${subfolder_list[$i]}" --step 7
  fi
done