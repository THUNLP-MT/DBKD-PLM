# DBKD-PLM
* Codebase for ACL 2023 conference long paper Bridging the Gap between Decision and Logits in Decision-based Knowledge Distillation for Pre-trained Language Models.
* Our code currently is based on huggingface and its run_glue scripts.
# Requirements
See `requirements.txt`
# Usage
To perform the proposed distillation method in our paper, please follow the following steps. We also include the example scripts in the `scripts/` folder.
* Step 1: Generate decision-to-logits look-up table before performing distillation by `python utils/decision2prob.py`, and it will generate a look-up table at `utils/ptable_mc10.pkl`.
* Step 2: Inference teacher model on a dataset by `bash scripts/inference_teacher.sh`.
  * If you do not have a teacher model, you can finetune one by `bash scripts/finetune_teacher.sh` before that.
  * If have your own teacher model, change the `TEACHER_PATH` variable to your path to teacher model.
* Step 3: Run knowledge distillation by `base scripts/distil.sh`. It will automatically do hyper-parameter search and evaluations. If you do not want to search hyper-parameter, we recommend to set `ALPHA=0.7`, `TEMP=1`, and `SIGMA=1`.
# Future plans
* If you have interests in developing some hybrid methods between our method and other baseline methods, you can explore the `hybrid` strategy in `student.py` :)
# Citation
```
ArXiv version comming soon...
```
