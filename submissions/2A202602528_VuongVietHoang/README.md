# DeepWeeds Lab Day 2 submission

Submission folder: `2A202602528_VuongVietHoang`.

## Reproducibility

- Notebook: `code/lab_day2_colab.ipynb`
- Platform: Google Colab with GPU; training is not intended to run locally.
- Dataset: DeepWeeds, original fold 0 CSV files.
- Python, PyTorch, timm, GPU model, and seeds: to be filled from the executed notebook output.
- Run cells in order. The notebook writes experiment logs/checkpoints to Google Drive and exports this submission directory after the runs finish.

## Submission contents

- `results.xlsx`: experiment tables defined by GUIDE.md section 6.1.
- `report.md`: conclusions based only on the executed experiment results.
- `curves/`: one training curve per experiment.
- `code/`: runnable Python modules, the unchanged `eval.py`, and the Colab notebook.
- `predictions/`: final and baseline test predictions for every final seed.

Do not submit the dataset or large model checkpoints. Preserve `exp_id` consistently across the workbook, curves, predictions, and report.
