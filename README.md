# Unifying Diversity and Fairness in Re-ranking via Economic Growth Theory

This repository provides an implementation and complete theorems of **DivFair**, a re-ranking method designed to unify **diversity** and **fairness** based on principles from economic growth theory. The framework can be applied as a post-processing module on top of recommendation systems.

Theorems and proof can be found at ```Theorems_and_Complete_Proof.pdf```.

---

## 🚀 Running the Code

To execute the re-ranking process for Steam dataset, you can run command:

```
python main.py --task recommendation --stage post-processing --dataset steam --train_config_file Post-processing.yaml
```
The hyper-parameters for DivFair are stored in ```properties/models/DivFair.yaml```.

The configuration for the re-ranking stage is stored in ```Post-processing.yaml```.

The ranking scores and test results can be found in ```log/```.
