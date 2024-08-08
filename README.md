# Usage

1. 修改synthesis_methods下对应方法的yaml配置文件
2. 

```bash
cd roles
export PYTHONPATH=../
python data_exchange.py [-h] [--data_type DATA_TYPE] [--dataset DATASET] [--dataset_path DATASET_PATH] [--dirichlet_alpha DIRICHLET_ALPHA] [--classes_per_client CLASSES_PER_CLIENT] [--balance BALANCE] [--n_provider N_PROVIDER] [--rounds ROUNDS]
                        [--consumer_model_name CONSUMER_MODEL_NAME] [--whether_resume {0,1}] [--resume_path RESUME_PATH] [--device_id DEVICE_ID] [--synthesis_method SYNTHESIS_METHOD] [--is_attack IS_ATTACK] [--n_attacker N_ATTACKER] [--consumer_batch_size CONSUMER_BATCH_SIZE]
                        [--num_eval NUM_EVAL] [--consumer_lr CONSUMER_LR] [--consumer_iterations CONSUMER_ITERATIONS] [--consumer_momentum CONSUMER_MOMENTUM] [--consumer_decay CONSUMER_DECAY]
```
