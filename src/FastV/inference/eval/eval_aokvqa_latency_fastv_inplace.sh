export CUDA_VISIBLE_DEVICES=0

model_path=/cpfs01/user/cl424408/models/llava-v1.5-13b
output_path=aokvqa_eval_fastv
mkdir -p $output_path

rank_list=(72 144 288 432) # rank equals to (1-R)*N_Image_Tokens, R=(87.5% 75% 50% 25%)
Ks=(2) 

for rank in ${rank_list[@]}; do
    for k in ${Ks[@]}; do
    # auto download the ocrvqa dataset
    python ./src/FastV/inference/eval/inference_aokvqa.py \
        --model-path $model_path \
        --use-fast-v \
        --fast-v-inplace \
        --fast-v-sys-length 36 \
        --fast-v-image-token-length 576 \
        --fast-v-attention-rank $rank \
        --fast-v-agg-layer $k \
        --output-path $output_path/aokvqa_13b_FASTV_4bit_inplace_${rank}_${k}.json 
    done
done


#Baseline
python ./src/FastV/inference/eval/inference_aokvqa.py \
    --model-path $model_path \
    --output-path $output_path/aokvqa_13b_baseline.json 
