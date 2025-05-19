from rdkit import Chem
from rdkit.Chem import Descriptors

file_path = "train.smiles"
molwts = []

with open(file_path, 'r') as f:
    for line in f:
        smi = line.strip()
        mol = Chem.MolFromSmiles(smi)
        if mol:
            molwts.append(Descriptors.MolWt(mol))

if molwts:
    avg_molwt = sum(molwts) / len(molwts)
    print(f"✅ 平均分子量为: {avg_molwt:.2f} Da")
else:
    print("❌ 无有效分子解析")