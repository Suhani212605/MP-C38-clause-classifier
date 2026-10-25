# MP-C38 Contract Clause Type Identifier (GPT-2 124M, LEDGAR / LexGLUE)
Student: Suhani S Rao (1GA23IS166), 7B, ISE, GAT Bengaluru. SDG 16 and SDG 8.

## Run (Google Colab with a T4 GPU is recommended; CPU works but is slow)
```
pip install -r requirements.txt
python data_preparation.py      # LEDGAR -> top-10 classes, 800/class, 70/15/15 split -> data/
python train.py                 # fine-tune, save models/best.pt, report_assets/results.json + charts
python evaluate.py              # re-evaluate saved checkpoint on the test set
python app.py                   # Gradio app (loads models/best.pt, never retrains)
pytest tests                    # verification tests
python fill_report.py LLM_Mini_project.docx MP-C38_Report.docx   # fills the report with real results
```
Options: `python train.py --epochs 4 --batch 16 --lr 5e-5 --max_len 256 --unfreeze 2 --seed 42`
(use `--batch 8` if GPU memory is low). Do not submit `models/best.pt` (weights not allowed).

## Files
`gpt_model.py` GPTModel + GPT-2 weight loading + new head | `utils.py` tokenizer, Dataset, PDF/TXT reading, clause splitting |
`data_preparation.py` | `train.py` | `evaluate.py` | `inference.py` (`clause_type(text)`) | `app.py` | `tests/` | `fill_report.py`.

## Demo inputs
"This Agreement shall be governed by the laws of the State of New York." -> expected a governing-law type;
"This Agreement may be executed in counterparts..." -> counterparts. Output: label, confidence, top-3.

## Status (honest)
Implemented: all files above. Tested here (no torch/network): clause splitting, CSV export, balancing/dedup/split integrity.
NOT yet executed: dataset download, GPT-2 weight loading, training, evaluation, tensor-shape tests, the Gradio app.
These need internet + torch (Colab). All report numbers are PENDING until you run them.
