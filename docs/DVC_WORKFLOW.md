# DVC Workflow Documentation

## Experiment 3: DVC Dataset Versioning and Git Integration

### Aim

This document explains the complete DVC workflow used in the MLOps Iris Classifier project for tracking dataset versions along with Git.

---

## DVC Remote Configuration

A local folder remote was configured as the default DVC remote.

```bash
dvc remote add -d myremote ~/dvc-remote-storage
```

This remote stores the actual dataset files outside the Git repository.

---

## Dataset Versioning Workflow

Every dataset change follows this workflow:

### 1. Add Dataset to DVC

```bash
dvc add data/raw/iris_v1.csv
```

DVC calculates an MD5 hash for the dataset and creates a lightweight `.dvc` metafile.

### 2. Stage DVC Metadata in Git

```bash
git add data/raw/iris_v1.csv.dvc data/raw/.gitignore
```

Git tracks only the `.dvc` pointer file and `.gitignore`, not the dataset itself.

### 3. Commit Dataset Version

```bash
git commit -m "data: add iris_v1 raw dataset (150 rows) tracked via DVC"
```

Each dataset version is recorded in Git history.

### 4. Push Dataset to DVC Remote

```bash
dvc push
```

The dataset is uploaded from the DVC cache to the configured DVC remote storage.

---

## Dataset Version History

### Version 1

- Dataset: `iris_v1.csv`
- Rows: **150**
- Tracked using DVC.

### Version 2

- Dataset: `iris_v1.csv`
- Rows: **170**
- Created by adding 20 synthetic rows.
- Updated using `dvc add` and committed to Git.

---

## Comparing Dataset Versions

To view dataset history:

```bash
git log --oneline -- data/raw/iris_v1.csv.dvc
```

To compare Version 1 with the current version:

```bash
dvc diff <commit_hash>
```

Output shows that `iris_v1.csv` was modified.

---

## Restoring an Older Dataset Version

### Restore Version 1 Pointer

```bash
git checkout <commit_hash> -- data/raw/iris_v1.csv.dvc
```

### Restore Actual Dataset

```bash
dvc checkout data/raw/iris_v1.csv.dvc
```

The dataset changes back to the historical version stored in the DVC cache.

### Restore Latest Version

```bash
git checkout HEAD -- data/raw/iris_v1.csv.dvc
dvc checkout data/raw/iris_v1.csv.dvc
```

This restores the latest dataset version.

---

## Git + DVC Workflow Summary

```
Dataset
   │
   ▼
dvc add
   │
   ▼
DVC Cache
   │
dvc push
   │
   ▼
DVC Remote Storage

.dvc metafile
   │
git add
   │
git commit
   │
git push
   ▼
GitHub Repository
```

---

## Outcome

- DVC successfully tracked multiple versions of the Iris dataset.
- Git stored only lightweight `.dvc` metadata files.
- The actual datasets were stored in DVC remote storage.
- Historical dataset versions were compared using `dvc diff` and restored using `dvc checkout`, ensuring reproducible machine learning experiments.