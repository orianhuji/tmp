#!/bin/zsh
#SBATCH --gres=gpu:1,vmem:40G
#SBATCH --mem=40g
#SBATCH -c2
#SBATCH --exclude=firth-01,cyril-01
#SBATCH --time=10:0:0
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=yuval.reif@mail.huji.ac.il
#SBATCH --output="/cs/labs/roys/yuval.reif/Tokens2Words/logs/slurm-%j.out"
#SBATCH --error="/cs/labs/roys/yuval.reif/Tokens2Words/logs/slurm-%j.out"

# Enable verbose mode
set -x

log_filename="slurm-${SLURM_JOB_ID}.out"
output_log="/cs/labs/roys/yuval.reif/Tokens2Words/logs/${log_filename}"

#export TRANSFORMERS_CACHE "/cs/snapless/roys/yuval.reif/cache/"
#export HF_HOME "/cs/snapless/roys/yuval.reif/cache/"

#module load cuda/12.4.1

source /cs/labs/roys/yuval.reif/envs/py3.11/bin/activate

#export TRANSFORMERS_CACHE="/cs/snapless/roys/yuval.reif/cache/"
export CUPY_CACHE_DIR="/cs/snapless/roys/yuval.reif/cache/.cupy/kernel_cache"
export TRANSFORMERS_CACHE="/cs/snapless/roys/lab_resources/"
export HF_HOME="/cs/snapless/roys/lab_resources/"
export HF_DATASETS_CACHE="/cs/snapless/roys/yuval.reif/cache/"
export HF_ACCELERATE_CACHE="/cs/snapless/roys/yuval.reif/cache/"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"

export HF_TOKEN=""

cd /cs/labs/roys/yuval.reif/Tokens2Words/src/

#model="Llama-3.1-8B"
model="Llama-2-7b-hf"
model_name="meta-llama/${model}"
output_dir="/cs/snapless/roys/yuval.reif/Tokens2Words/runs/vocab_expansion/${model}/"

dataset="wikitext"
data_split="test"

#calibration_dataset="wikitext"
#calibration_split="train"
calibration_dataset="pg19"
calibration_split="validation"
calibration_lr="0.0001"
calibration_num_epochs=5

#dataset="pg19"
#data_split="validation"

#dataset="fineweb-edu"
#data_split="train"

#detokenization_decision_rule="1st_id_layer"
detokenization_decision_rule="2nd_id_layer"
detokenization_decision_rule_E="1st_id_layer"
detokenization_max_valid_layer="18"
min_freq="5"

procrustes_layer="all"
translators_name="procrustes_v2_rms_all_layers_extract_x"

#exp_name="${dataset}/full/${data_split}/${translators_name}_patchscopes_x_x_x_x_${detokenization_decision_rule}"
#exp_name="${dataset}/full/${data_split}/${translators_name}_min_freq_${min_freq}_patchscopes_x_x_x_x_${detokenization_decision_rule}_and_E_${detokenization_decision_rule_E}_max_l${detokenization_max_valid_layer}"
exp_name="${dataset}/full/${data_split}/calibration_E_and_U_on_${calibration_dataset}_${calibration_split}_${calibration_num_epochs}epochs_lr${calibration_lr}_${translators_name}_min_freq_${min_freq}_patchscopes_x_x_x_x_${detokenization_decision_rule}_and_E_${detokenization_decision_rule_E}_max_l${detokenization_max_valid_layer}"
#exp_name="${dataset}/full/${data_split}/rescale_logits_${translators_name}_min_freq_${min_freq}_patchscopes_x_x_x_x_${detokenization_decision_rule}_and_E_${detokenization_decision_rule_E}_max_l${detokenization_max_valid_layer}"

patchscopes_prompt="X X X X"
#patchscopes_prompt="X, X, X, X,"

#extraction_prompt="\n\n\n\n\n\n\n\nX"
extraction_prompt="X"
#extraction_prompt="X X X X"

translators_path="${output_dir}/translators/v2/${translators_name}.pt"
#translators_path="${output_dir}/${exp_name}/translators.pt"

#patchscopes_cache="${output_dir}/${exp_name}/patchscopes_results.parquet"
patchscopes_cache="${output_dir}/patchscopes/${dataset}/prompt_x_x_x_x.parquet"



mkdir -p ${output_dir}
output_log_link="${output_dir}/${log_filename}"
ln -s "${output_log}" "${output_log_link}"

# --words_filter_min_freq 10

#python -m tokens2words.run_vocab_expansion_eval --output_dir "${output_dir}" --words_filter_min_freq "${min_freq}"  --words_filter_non_en --patchscopes_results_cache "${patchscopes_cache}" --translators_path "${translators_path}" --words_dataset "${dataset}" --eval_dataset "${dataset}" --exp_name "${exp_name}" --extraction_prompt "${extraction_prompt}" --patchscopes_prompt "${patchscopes_prompt}" --detokenization_max_valid_layer ${detokenization_max_valid_layer} --detokenization_decision_rule "${detokenization_decision_rule}" --detokenization_decision_rule_E "${detokenization_decision_rule_E}" --model_name "${model_name}" --extraction_batch_size 32   --eval_dataset_split "${data_split}" --words_dataset_split "${data_split}"

# calibration:
python -m tokens2words.run_vocab_expansion_eval --output_dir "${output_dir}" --calibrate_new_lm_head  --calibration_lr "${calibration_lr}" --calibration_dataset "${calibration_dataset}" --calibration_dataset_split "${calibration_split}" --calibration_num_epochs "${calibration_num_epochs}"  --words_filter_min_freq "${min_freq}"  --words_filter_non_en --patchscopes_results_cache "${patchscopes_cache}" --translators_path "${translators_path}" --words_dataset "${dataset}" --eval_dataset "${dataset}" --exp_name "${exp_name}" --extraction_prompt "${extraction_prompt}" --patchscopes_prompt "${patchscopes_prompt}" --detokenization_max_valid_layer ${detokenization_max_valid_layer} --detokenization_decision_rule "${detokenization_decision_rule}" --detokenization_decision_rule_E "${detokenization_decision_rule_E}" --model_name "${model_name}" --extraction_batch_size 32   --eval_dataset_split "${data_split}" --words_dataset_split "${data_split}"


rm -rf "${output_log_link}"
cp "${output_log}" "${output_log_link}"
rm -rf "${output_log}"


