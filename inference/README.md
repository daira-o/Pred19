# Private inference files

The local preparation workflow writes `inference_input.csv` into this folder.

`demo_synthetic.csv` is a separate, fully synthetic dataset committed for the public Streamlit demonstration.
It contains no real patient observations and is loaded with the **Synthetic dataset** toggle.

The project works with clinical observations originating from German hospitals. Do not commit
`inference_input.csv` unless its public release has been explicitly authorized; it is intentionally ignored by
Git. To use a local inference file, open **Patient Monitor**, disable **Synthetic dataset**, and upload the CSV.
The application processes the upload in the active session.

Only the six model inputs are expected:

```text
PCR,LDH,WBC,CA,HCT,EO
```
