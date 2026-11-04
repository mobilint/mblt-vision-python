# Test `vision`

You can validate Mobilint's Vision API with [`pytest`](https://docs.pytest.org/en/stable/). The snippets below assume your virtual environment is already activated.

## Install Packages

Install the runtime extras plus the developer tooling required by the test suite:

```bash
pip install -e .
pip install pytest
```

## Run All Tests

Execute the complete standalone Vision test matrix:

```bash
pytest tests
```

## Run Offline Unit Tests

Exclude Hugging Face downloads and NPU hardware:

```bash
pytest tests -m "not requires_network and not requires_npu"
```

## Run Optional Integration Tests

After authenticating with Hugging Face Hub, exercise representative ONNX models:

```bash
pytest tests/test_onnx_classification.py -m requires_network
```

To check representative Hub configurations without downloading model bytes or
using an inference backend:

```bash
pytest tests/test_hub_validation.py -m requires_network
```

These read-only checks use the exact configured `repo_id` and `revision`, verify
MXQ under the normalized target board and ONNX at the repository root, and pin
metadata listing to the resolved commit to avoid races with moving branches.
Local-only artifacts are skipped without a network request. An observed current
commit is diagnostic metadata, not evidence of a historically approved revision;
missing approval provenance must never be replaced with today's `main` tip.
Offline mocked tests cover missing files, wrong boards, API failures and local-only
restrictions. No YAML revision or model artifact is changed by these checks.

To run MXQ inference, add a configured NPU and the shared runtime options:

```bash
pytest tests/test_mxq_inference.py -m requires_npu --mxq-path /path/to/model.mxq
```
