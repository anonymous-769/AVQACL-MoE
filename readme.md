# AVQACL-MoE: Anchor-Based Mixture-of-Experts for Audio-Visual Question Answering Continual Learning

🏠 Homepage | 🤗 Weights (Including Experts weights and anchors) | 📖 Paper (To Appear)

AVQACL-MoE is an anchor-based Mixture-of-Experts (MoE) framework for Audio-Visual Question Answering Continual Learning (AVQACL). It freezes a pre-trained audio-visual backbone, incrementally trains task-specialized experts, and performs task-independent routing via modality-specific anchors at inference.

- Reduces forgetting from 27% to 2% and improves final accuracy by over 30.9% on AVQACL.
- Task-agnostic inference via cosine similarity between inputs and learned anchors.
- Script-driven training/evaluation for easy reproduction.


## 📂 Open Resources
- Code: this repository
- Data jsons: `data_sample/`
- Training Script: `train_all_tasks.sh`
- Evaluation Script: `eval_all_tasks.sh`


## ⚙️ Environment
- Use `environment.yml` to create the environment
  - `conda env create -f environment.yml`
  - `conda activate avqacl-moe`
- Minimal packages reference: `env.txt`


## 📦 Checkpoints (must prepare before training)
Place the following at the exact paths:
- [`checkpoints/AVQACL-MoE-base`](https://huggingface.co/anonymous-769/AVQACL-MoE)
- [`checkpoints/clip-vit-large-patch14-336`](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main)
- [`checkpoints/BEATs_iter3_plus_AS2M.pt`](https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf)


## 🗂️ Data Preparation
- Quick start: use `data_sample/` to verify end-to-end pipeline.
- Full/custom data: follow the JSON format in `data_sample/` (e.g., `train_*.json`, `test_*.json`).
- Ensure file names referenced in the scripts exist. If your dataset uses different names, edit the script paths accordingly before running.


## 🚀 Quick Start
Step 0 — Environment
- Create and activate the Conda env as above.

Step 1 — Prepare Checkpoints
- Make sure the three checkpoints are placed under `checkpoints/` with the exact names.

Step 2 — Verify/Prepare Data
- Keep your train/val/test JSON files consistent with script references.

Step 3 — Train (script-based)
- Run: `bash train_all_tasks.sh` (in a bash-capable environment: Linux/cluster/WSL)
- What happens: sequentially trains experts for tasks 0→3 in anchor mode, freezing the backbone and saving experts/anchors and logs (see Outputs).
- We also provide pre-trained experts and anchors for reproducing the split-AVQA and split-AVQA-MUSIC dataset results so that you don't need to run training scripts:[`checkpoints/AVQACL-MoE-experts`](https://huggingface.co/anonymous-769/AVQACL-MoE-expert)

Step 4 — Evaluate (script-based)
- Run: `bash eval_all_tasks.sh`
- What happens: loads `output/AVQA/experts` and `output/AVQA/anchors`, evaluates tasks 0–3, and saves results JSONs and logs (see Outputs).
- We also provide pre-trained experts and anchors for reproducing the split-AVQA and split-AVQA-MUSIC dataset results:[`checkpoints/AVQACL-MoE-experts`](https://huggingface.co/anonymous-769/AVQACL-MoE-expert)

Step 5 — Locate Results
- See `eval_results/AVQA/` for per-task JSONs and `logs/` for progress and summaries.


## 📤 Outputs (after successful training/evaluation)
Training artifacts
- Experts: `output/AVQA/experts/expert_task_{0..3}.pt`
- Anchors: `output/AVQA/anchors/anchor_task_{0..3}.pt`
- Logs: `logs/train_all_tasks.log`, `logs/train_all_tasks.err`

Evaluation artifacts
- Results:
  - `eval_results/AVQA/task0_come_results.json`
  - `eval_results/AVQA/task1_happening_results.json`
  - `eval_results/AVQA/task2_where_results.json`
  - `eval_results/AVQA/task3_which_results.json`
- Logs: `logs/eval_all_tasks.log`, `logs/eval_all_tasks.err`


## 🧱 Architecture (high level)
- Frozen audio-visual backbone
- Task-specialized experts (one per task) trained incrementally
- Modality-specific anchors learned during training
- Task-independent routing via cosine similarity between input features and anchors


## 🧪 Tasks (example sequence 0→3 of AVQA dataset)
- Task 0: Come
- Task 1: Happening
- Task 2: Where
- Task 3: Which


## 📁 Project Structure (partial)
```
AVQACL-MoE/
├── model/
│   ├── moe/                  # Anchor-based MoE layers and routing
│   ├── language_model/
│   └── ...
├── train/
│   ├── train_anchor_moe.py   # Anchor training entry
│   └── ...
├── eval_anchor.py            # Anchor-based evaluation
└── ...
```


## 🔁 Reproducibility Checklist
- Code release: full training/evaluation pipeline in this repo.
- Data: samples included; prepare full AVQACL data in the same JSON format.
- Environment: `environment.yml` provided; please report CUDA/PyTorch versions in issues for best support.
- Hardware: validated on A100 NVIDIA GPUs; A100 recommended for our configs.


## 📊 Results (Summary)
- On AVQACL, AVQACL-MoE reduces forgetting from 27% to 2% and improves final accuracy by over 30.9% under continual learning.
- For detailed results and ablations, please refer to the paper.


## 📝 Citation
If you find our work useful, please cite:
```bibtex
@inproceedings{avqacl_moe,
  title     = {AVQACL-MoE: Anchor-Based Mixture-of-Experts for Audio-Visual Question Answering Continual Learning},
  author    = {To be updated},
  booktitle = {To be updated},
  year      = {2026}
}
```
