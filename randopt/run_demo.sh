#!/usr/bin/env bash

# RandOpt YouTube Demo Çalıştırma Betiği
# Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights

# Ortamı ve değişkenleri ayarla
cd RandOpt_repo
export CUDA_VISIBLE_DEVICES="0"  # Eğer birden fazla GPU varsa "0,1" gibi güncelleyin
export VLLM_NO_USAGE_STATS=1

echo "=========================================================="
echo "          RandOpt (Neural Thickets) Model Eğitimi         "
echo "=========================================================="
echo "Veri Seti: tutorial_math"
echo "Hedef Model: Qwen/Qwen2.5-1.5B-Instruct"
echo ""

# RandOpt argümanlarını kullanarak otonom ensemble eğitimini başlatın.
# Paper'da bahsedilen RandOpt mekanizması:
# 1. 50 popülasyon yaratılır (-population_size).
# 2. Her birinde modelek rastgele gürültü eklenir (-sigma_values).
# 3. Model bu mini-dataset'te skor alır ve en iyi %10 (top_k_ratios 0.1) oylanır.

python3 randopt.py \
  --dataset tutorial_math \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --num_engines 1 \
  --tp 1 \
  --precision bfloat16 \
  --population_size 50 \
  --top_k_ratios "0.1" \
  --sigma_values "0.0005,0.001" \
  --max_tokens 256 \
  --global_seed 42 \
  --experiment_dir "randopt_tutorial_results" 

echo "=========================================================="
echo " Eğitim (Gürültü Optimizasyon) Tamamlandı!"
echo " Sonuçları RandOpt/randopt_tutorial_results klasöründe bulabilirsiniz."
echo "=========================================================="
