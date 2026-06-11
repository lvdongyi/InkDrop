# InkDrop: Invisible Backdoor Attacks Against Dataset Condensation
[![arXiv](https://img.shields.io/badge/arXiv-2603.28092-red.svg)](https://arxiv.org/abs/2603.28092)

## ⚠️ Authorship Dispute Notice / 署名争议声明

This repository contains the implementation code for *InkDrop: Invisible Backdoor Attacks Against Dataset Condensation*.

Upon resubmission to IEEE Transactions on Dependable and Secure Computing (TDSC), my name was removed from the author list without my consent. I formally raised this matter with the Editor-in-Chief of TDSC, who suspended the review process. Per subsequent editor correspondence, the submission has since been withdrawn by the co-author (He Yang). The underlying authorship dispute itself remains unresolved.

This notice is made to ensure transparent attribution of this work.

![Editor correspondence](docs/editor_feedback.png)

---

本仓库为论文 *InkDrop: Invisible Backdoor Attacks Against Dataset Condensation* 的实现代码。

在向 IEEE Transactions on Dependable and Secure Computing（TDSC）重新投稿时，本人姓名在未经本人同意的情况下被**杨和**从作者列表中移除。本人已就此事正式向 TDSC 主编反映，主编已暂停审稿流程。据编辑后续反馈，该投稿已由**杨和**撤回。但署名争议本身尚未得到解决。

本人公开发布此声明，旨在确保本工作的署名透明。

---

## Usage
1. 修改 `synthesis_methods` 下对应方法的 yaml 配置文件
2. 
\```bash
cd roles
export PYTHONPATH=../
python data_exchange.py [-h] [--data_type DATA_TYPE] [--dataset DATASET] [--dataset_path DATASET_PATH] [--dirichlet_alpha DIRICHLET_ALPHA] [--classes_per_client CLASSES_PER_CLIENT] [--balance BALANCE] [--n_provider N_PROVIDER] [--rounds ROUNDS]
                        [--consumer_model_name CONSUMER_MODEL_NAME] [--whether_resume {0,1}] [--resume_path RESUME_PATH] [--device_id DEVICE_ID] [--synthesis_method SYNTHESIS_METHOD] [--is_attack IS_ATTACK] [--n_attacker N_ATTACKER] [--consumer_batch_size CONSUMER_BATCH_SIZE]
                        [--num_eval NUM_EVAL] [--consumer_lr CONSUMER_LR] [--consumer_iterations CONSUMER_ITERATIONS] [--consumer_momentum CONSUMER_MOMENTUM] [--consumer_decay CONSUMER_DECAY]
\```

## Examples
\```bash
nohup python -u data_exchange.py --device_id 3 --consumer_model_name AlexNet --backdoor_method casev2 > log.txt &
\```
