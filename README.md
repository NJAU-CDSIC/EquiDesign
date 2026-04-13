# **EquiDesign** Code Repository

This is a directory for storing the **EquiDesign** model code and data. **EquiDesign** is a deep learning framework for protein sequence design from backbone structures.

---

The folders in the EquiDesign repository:

## **Datasets**

a. **CATH4.4**: A non-redundant protein structure dataset from the CATH 4.4 database.  
b. **TS50**: A standard benchmark dataset consisting of 50 protein structures.  
c. **TS500**: A large-scale benchmark dataset containing 500 protein structures.

## **EquiDesign_code**

Main code files for the EquiDesign model.

- `Model`: model implementation  
- `Model_training`: training entry  
- `Model_testing`: evaluation scripts

## **SOTA**

Comparative methods used in the contrast experiments (see `SOTA/README.md`).

## **Scripts**

Contains auxiliary scripts for data processing and evaluation.

- `Scripts/build_chain_set_from_cath44.py`: build `chain_set.jsonl` and `chain_set_splits.json` from CATH v4.4.0 raw files  
- `Scripts/stability/`: Hybrid Score 3 evaluation

---

## **Step-by-step Running**

### 1. Environment Installation

It is recommended to use a conda environment (Python 3.10). See `environment.yml` for details.

```bash
conda env create -f environment.yml
conda activate equidesign
```

### 2. Datasets

Place the datasets into the `Datasets` folder.

For CATH4.4, you can download raw files by running:

```bash
cd Datasets/CATH4.4
bash getCATH.sh
```

Then build `chain_set.jsonl` and `chain_set_splits.json`:

```bash
python ../../Scripts/build_chain_set_from_cath44.py --list_file cath-dataset-nonredundant-S40-v4_4_0.list --pdb_dir ./pdb --out_dir .
```

### 3. Training and Testing

Train the model using the CATH4.4 dataset:

```bash
bash EquiDesign_code/Model_training/training.sh
```

Test on TS50:

```bash
bash EquiDesign_code/Model_testing/test_50.sh
```

Test on TS500:

```bash
bash EquiDesign_code/Model_testing/test_500.sh
```

---

## 4. Installation

```bash
git clone <YOUR_GITHUB_REPO_URL>
```

## 5. Citation

If you use EquiDesign in your research, please cite:

```bibtex
@misc{EquiDesign2026,
  title={EquiDesign: Protein Sequence Design Based on EquiFormer},
  author={NJAU-CDSIC},
  year={2026},
  howpublished={\\url{<YOUR_GITHUB_REPO_URL>}}
}
```

