# **EquiDesign** Code Repository

This is a directory for storing the **EquiDesign** model code and data. **EquiDesign** is a deep learning framework for protein sequence design from backbone structures.

---

The folders in the EquiDesign repository:

- **Datasets**:
  
	a. **CATH4.4**: A non-redundant protein structure dataset from the CATH 4.4 database.

	b. **TS50**: A standard benchmark dataset consisting of 50 protein structures.

	c. **TS500**: A large-scale benchmark dataset containing 500 protein structures.  

- **EquiDesign_code**: Main code file for the EquiDesign model.

- **SOTA**:Comparative methods used in the contrast experiments:

	SPROF: https://github.com/biomed-AI/SPROF

	ProDCoNN: https://github.com/wells-wood-research/timed-design

	GraphTrans:https://github.com/jingraham/neurips19-graph-protein-design

	GVP: https://github.com/drorlab/gvp

	GCA: https://github.com/chengtan9907/gca-generative-protein-design

	AlphaDesign: https://github.com/Westlake-drug-discovery/AlphaDesign

	ProteinMPNN: https://github.com/dauparas/ProteinMPNN

	Frame2seq: https://github.com/dakpinaroglu/Frame2seq

	PiFold: https://github.com/A4Bio/PiFold

	GeoSeqBuilder: https://github.com/PKUliujl/GeoSeqBuilder

	ProtSeqGen: https://github.com/NJAU-CDSIC/ProtSeqGen

- **Scripts**: Contains auxiliary scripts for data processing and visualization.

---

## **Step-by-step Running**

### 1. Environment Installation

- It is recommended to use a conda environment (Python 3.10).
- See `environment.yml` for details.

```bash
conda env create -f environment.yml
conda activate equidesign
```

### 2. Datasets

- For CATH4.4, you can download raw files by running:

```bash
bash getCATH.sh
```

- Then build `chain_set.jsonl` and `chain_set_splits.json`:

```bash
python ../../Scripts/build_chain_set_from_cath44.py --list_file cath-dataset-nonredundant-S40-v4_4_0.list --pdb_dir ./pdb --out_dir .
```

### 3. Training and Testing

- Train the model using the CATH4.4 dataset:

```bash
bash EquiDesign_code/Model_training/training.sh
```

- Test on TS50:

```bash
bash EquiDesign_code/Model_testing/test_50.sh
```

- Test on TS500:

```bash
bash EquiDesign_code/Model_testing/test_500.sh
```

---

## 4. Scripts Usage

- **Training example **

```bash
bash train_equidesign_cath44_example.sh
```

- **Hybrid Score 3 (Stability)**


```bash
python eval_design_hybrid_score.py all \
  --jsonl_file ../../Datasets/CATH4.4/chain_set.jsonl \
  --split_file ../../Datasets/CATH4.4/chain_set_splits.json \
  --checkpoint ../../EquiDesign_code/Model/model_weights/best_model.pt \
  --out_csv ./out/design_with_hybrid.csv
```

---

## 5. Installation

```bash
git clone https://github.com/NJAU-CDSIC/EquiDesign.git
```

## 6. Citation

If you use EquiDesign in your research, please cite:

```bibtex
@misc{EquiDesign2026,
  title={EquiDesign: Protein Sequence Design Based on EquiFormer},
  author={NJAU-CDSIC},
  year={2026},
  howpublished={\\url{https://github.com/NJAU-CDSIC/EquiDesign}}
}
```
