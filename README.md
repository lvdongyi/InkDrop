# InkDrop: Invisible Backdoor Attacks Against Dataset Condensation

## ⚠️ Authorship Dispute Notice / 署名争议声明

This repository contains the implementation code for *InkDrop: Invisible Backdoor Attacks Against Dataset Condensation*.

Upon resubmission to IEEE Transactions on Dependable and Secure Computing (TDSC), my name was removed from the author list **without my knowledge or consent**, by a co-author (surnamed **Yang**) affiliated with Xi'an Jiaotong University. I have formally raised this issue with the Editor-in-Chief of TDSC, who has **suspended the review process** pending resolution. As of the date of this notice, the dispute remains unresolved due to the co-author's lack of response.

This notice is made public to ensure transparent attribution of this work and to protect my legitimate academic rights.

---


本仓库为论文 *InkDrop: Invisible Backdoor Attacks Against Dataset Condensation* 的实现代码。

在向 IEEE Transactions on Dependable and Secure Computing（TDSC）重新投稿时，本人姓名在**未经本人知情与同意的情况下**被从作者列表中移除，实施此操作的为西安交通大学某**杨姓**合著者。本人已就此事正式向 TDSC 主编反映，主编已**暂停审稿流程**待问题解决。截至本声明发布之日，由于该合著者未予回应，争议仍未解决。

本人公开发布此声明，旨在确保本工作的署名透明，并维护本人合法的学术权益。

---

## Implementation of "InkDrop: Invisible Backdoor Attacks Against Dataset Condensation"

## Usage
1. 修改 `synthesis_methods` 下对应方法的 yaml 配置文件
2. 
```bash
cd roles
export PYTHONPATH=../
python data_exchange.py [-h] [--data_type DATA_TYPE] [--dataset DATASET] [--dataset_path DATASET_PATH] [--dirichlet_alpha DIRICHLET_ALPHA] [--classes_per_client CLASSES_PER_CLIENT] [--balance BALANCE] [--n_provider N_PROVIDER] [--rounds ROUNDS]
                        [--consumer_model_name CONSUMER_MODEL_NAME] [--whether_resume {0,1}] [--resume_path RESUME_PATH] [--device_id DEVICE_ID] [--synthesis_method SYNTHESIS_METHOD] [--is_attack IS_ATTACK] [--n_attacker N_ATTACKER] [--consumer_batch_size CONSUMER_BATCH_SIZE]
                        [--num_eval NUM_EVAL] [--consumer_lr CONSUMER_LR] [--consumer_iterations CONSUMER_ITERATIONS] [--consumer_momentum CONSUMER_MOMENTUM] [--consumer_decay CONSUMER_DECAY]
```

## Examples
```bash
nohup python -u data_exchange.py --device_id 3 --consumer_model_name AlexNet --backdoor_method casev2 > log.txt &
```
