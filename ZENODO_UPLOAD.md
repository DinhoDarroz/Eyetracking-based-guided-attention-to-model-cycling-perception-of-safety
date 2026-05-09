# Zenodo Upload Checklist

This dataset should be uploaded to Zenodo as a **dataset record**, while GitHub
keeps the source code, documentation, and release scripts.

## What Goes Where

- GitHub:
  - Training/evaluation code.
  - Dataset documentation.
  - Release preparation and validation scripts.
  - Small metadata files such as `CITATION.cff`, `DATASET.md`,
    `DATASET_CARD.md`, `data_dictionary.csv`, and `zenodo_metadata.json`.

- Zenodo:
  - `EG-PCS-Dataset-v1.0.0.tar.gz`
  - The large images, gaze maps, CSV/Parquet/pickle comparison files, checksums,
    and release docs inside that archive.
  - The DOI researchers should cite for the dataset.

GitHub links to Zenodo through the DOI. Zenodo links back to GitHub through the
related software repository URL in `zenodo_metadata.json`.

## Prepared Local Files

- Release folder: `EG-PCS-Dataset-v1.0.0/`
- Upload archive: `EG-PCS-Dataset-v1.0.0.tar.gz`
- Reserved DOI: `10.5281/zenodo.20101496`
- Archive SHA-256:

```text
ae3d5f81f342d56b7f96de76ff892ab77a0f31ace0eeaad9a596ff1b5a11106c  EG-PCS-Dataset-v1.0.0.tar.gz
```

## Manual Zenodo Upload

1. Go to <https://zenodo.org/uploads>.
2. Create a new upload.
3. Select resource type: `Dataset`.
4. Upload `EG-PCS-Dataset-v1.0.0.tar.gz`.
5. Fill metadata using `zenodo_metadata.json`.
6. In the DOI field, choose the option to reserve/get a DOI.
7. Copy the reserved DOI into `CITATION.cff`.
8. Rebuild the release folder/archive so the final archive contains the DOI.
9. Upload the rebuilt archive to the same draft.
10. Review all metadata and files.
11. Publish the Zenodo record.
12. Add the final DOI link to `README.md`.

## API Upload

Create a Zenodo personal access token with deposit write permissions, then set it
in your shell. Do not commit or paste the token into the repository.

```bash
export ZENODO_TOKEN="..."
```

Create a draft and upload the archive:

```bash
python scripts/upload_dataset_to_zenodo.py \
  EG-PCS-Dataset-v1.0.0.tar.gz \
  --metadata zenodo_metadata.json
```

The script does not publish by default. It prints the draft URL and any reserved
DOI returned by Zenodo. Review the draft in the browser before publishing.

If you reserve a DOI, update the local metadata, and rebuild the archive, replace
the file in the existing draft:

```bash
python scripts/update_zenodo_draft_file.py \
  20101496 \
  EG-PCS-Dataset-v1.0.0.tar.gz
```

For a test run, use the Zenodo sandbox:

```bash
python scripts/upload_dataset_to_zenodo.py \
  EG-PCS-Dataset-v1.0.0.tar.gz \
  --metadata zenodo_metadata.json \
  --sandbox
```
