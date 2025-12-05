# ChemQwen: Computational Chemistry AI Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)

## 🧪 About
ChemQwen is a specialized AI model fine-tuned on computational chemistry knowledge. Built on the Qwen3-0.6B base model using LoRA (Low-Rank Adaptation) fine-tuning, this model provides expert-level explanations for computational chemistry concepts including:

- Density Functional Theory (DFT) calculations
- Molecular dynamics simulations  
- Thermodynamic property calculations (Gibbs free energy, entropy, etc.)
- Quantum chemistry methods and basis sets
- Solvation models and solvent effects

## 🤗 Model Access
The trained model is hosted on Hugging Face Hub:  
https://huggingface.co/Elham0815/chemqwen

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Elham0815/chemqwen.git
cd chemqwen
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
