export CUDA_VISIBLE_DEVICES=0

model_path=./models/llava-v1.5-7b
output_path=ocrvqa_eval_fastv
mkdir -p $output_path

rank_list=(72 144 288 432) # rank equals to (1-R)*N_Image_Tokens, R=(75% 50% 25% 12.5%)
Ks=(2) 

for rank in ${rank_list[@]}; do
    for k in ${Ks[@]}; do
    # auto download the ocrvqa dataset
    python ./src/FastV/inference/eval/inference_ocrvqa.py \
        --model-path $model_path \
        --use-fast-v True \
        --fast-v-sys-length 36 \
        --fast-v-image-token-length 576 \
        --fast-v-attention-rank $rank \
        --fast-v-agg-layer $k \
        --output-path $output_path/ocrvqa_7b_FASTV_nocache_mt40_${rank}_${k}.json 
    done
done
