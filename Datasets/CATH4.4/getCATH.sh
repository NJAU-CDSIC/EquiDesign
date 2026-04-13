

# Download CATH v4.4.0 release files (the first two dirs under v4_4_0):

#

# 1) cath-classification-data/ (metadata; optional for training, useful for mapping/analysis)

#   http://download.cathdb.info/cath/releases/all-releases/v4_4_0/cath-classification-data/
#

# 2) non-redundant-data-sets/ (structures; required to build chain_set.jsonl)

#   http://download.cathdb.info/cath/releases/all-releases/v4_4_0/non-redundant-data-sets/
#

# After downloading + unpacking, build `chain_set.jsonl` / `chain_set_splits.json` with:

#   python ../../Scripts/build_chain_set_from_cath44.py \
#     --list_file cath-dataset-nonredundant-S40-v4_4_0.list \
#     --pdb_dir ./pdb \
#     --out_dir .

echo "[1/2] Downloading CATH classification metadata..."
wget -nc http://download.cathdb.info/cath/releases/all-releases/v4_4_0/cath-classification-data/README-cath-domain-boundaries-file-format.txt
wget -nc http://download.cathdb.info/cath/releases/all-releases/v4_4_0/cath-classification-data/cath-domain-boundaries-v4_4_0.txt
echo "(Note) cath-domain-list-S40-v4_4_0.txt may not exist; skipping if unavailable."
wget -nc http://download.cathdb.info/cath/releases/all-releases/v4_4_0/cath-classification-data/cath-domain-list-S40-v4_4_0.txt || true

echo "[2/2] Downloading CATH non-redundant S40 structures..."
wget -nc http://download.cathdb.info/cath/releases/all-releases/v4_4_0/non-redundant-data-sets/cath-dataset-nonredundant-S40-v4_4_0.list
wget -nc http://download.cathdb.info/cath/releases/all-releases/v4_4_0/non-redundant-data-sets/cath-dataset-nonredundant-S40-v4_4_0.pdb.tgz

mkdir -p pdb
tar -xzf cath-dataset-nonredundant-S40-v4_4_0.pdb.tgz -C pdb

echo ""
echo "[OK] Downloaded and unpacked CATH S40 PDBs to: $(pwd)/pdb"
echo "Next step (build jsonl/splits):"
echo "  python ../../Scripts/build_chain_set_from_cath44.py --list_file cath-dataset-nonredundant-S40-v4_4_0.list --pdb_dir ./pdb --out_dir ."
echo ""

