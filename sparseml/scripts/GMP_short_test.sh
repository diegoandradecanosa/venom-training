#!/bin/bash

export workdir=../
# Setting workdir to have known path for referencing different parts of the repository.
#export RECIPE_NAME=bert-base-12layers_prune90-GMP.yaml
export RECIPE_NAME=GMP_short_test.yaml
export RECIPE=./recipes/${RECIPE_NAME}
export EXEC_ID=GMP_short_test

#../integrations/huggingface-transformers/recipes/bert-base-12layers_prune90-GMP.md

# for 12-layer model: export MODEL=bert-base-uncased
# for 6-layer model: export MODEL=neuralmagic/oBERT-6-upstream-pretrained-dense
# for 3-layer model: export MODEL=neuralmagic/oBERT-3-upstream-pretrained-dense
#export MODEL=bert-base-uncased

#debug: -m pdb

#python transformers/examples/pytorch/question-answering/run_qa.py \
python $workdir/src/sparseml/transformers/question_answering.py \
  --distill_teacher neuralmagic/oBERT-teacher-squadv1 \
  --model_name_or_path bert-large-uncased \
  --dataset_name squad \
  --do_train \
  --fp16 \
  --do_eval \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $workdir/integrations/huggingface-transformers/MODELS_DIR/$EXEC_ID \
  --overwrite_output_dir \
  --cache_dir cache \
  --preprocessing_num_workers 6 \
  --seed 42 \
  --num_train_epochs 50 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --recipe $RECIPE \
  --max_train_samples 2000 \
  --max_eval_samples 2000 
 # --report_to wandb


# Segunda ejecución

 #export RECIPE_NAME=GMP-64:X:10_50epochs.yaml
 #export RECIPE=$workdir/integrations/huggingface-transformers/recipes/${RECIPE_NAME}
 #export EXEC_ID=64:X:10_50epochs:V2

 #python $workdir/src/sparseml/transformers/question_answering.py \
 #  --distill_teacher neuralmagic/oBERT-teacher-squadv1 \
 #  --model_name_or_path bert-base-uncased \
 #  --dataset_name squad \
 #  --do_train \
 #  --fp16 \
 #  --do_eval \
 #  --evaluation_strategy epoch \
 #  --per_device_train_batch_size 16 \
 #  --learning_rate 5e-5 \
 #  --max_seq_length 384 \
 #  --doc_stride 128 \
 #  --output_dir $workdir/integrations/huggingface-transformers/MODELS_DIR/$EXEC_ID \
 #  --overwrite_output_dir \
 #  --cache_dir cache \
 #  --preprocessing_num_workers 6 \
 #  --seed 42 \
 #  --num_train_epochs 50 \
 #  --save_steps 1000 \
 #  --save_total_limit 2 \
 #  --recipe $RECIPE
