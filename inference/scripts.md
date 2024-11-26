# Thực hiện retrieval cho model sau round 1
bash inference/encode_model.sh pretrained_model/model_repllama_50_hard_round1_2_batch/checkpoint-2800 temp/jp_round1_2800

# Thực hiện retrieval cho model sau round 2
bash inference/encode_model.sh pretrained_model/model_repllama_50_hard_round2_1_BATCH_THOI/checkpoint-400 temp/jp_full_cp_ckpt_round2_400