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

To run MXQ inference, add a configured NPU and the shared runtime options:

```bash
pytest tests/test_mxq_inference.py -m requires_npu --mxq-path /path/to/model.mxq
```
