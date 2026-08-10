---
name: mblt-vision-readme
description: >-
  Write and maintain mblt-vision-python README documentation, API examples, model references,
  and migration notes.
---

# Mobilint Vision README Writing

## Documentation Ownership

- Keep the root README concise: package purpose, installation, a minimal example, and a link to
  mblt_vision/README.md.
- Keep the detailed Vision API reference in mblt_vision/README.md. It owns Python construction,
  framework selection, model discovery, model-family tables, output taxonomy, and migration notes.
- Document Model Zoo compatibility as migration context only. Do not present Model Zoo CLI,
  validation, dataset organization, or compilation commands as features of this package.

## Accuracy Rules

- Use the public mblt_vision namespace in every executable example.
- Use model_path for new local-artifact examples. Mention mxq_path and onnx_path only as
  compatibility aliases.
- State that .mxq and .onnx paths select their framework automatically when framework is omitted.
  Document the explicit-framework conflict error.
- Describe file_cfg.filename as the MXQ source artifact and same-stem ONNX derivation. Mention
  onnx_filename only for a genuinely different published artifact.
- Keep post_cfg.dataset terminology precise: it identifies output taxonomy, not just a task.
- Use obb as the only oriented-bounding-box name in standalone documentation.

## Style and Validation

- Use ATX headings, one blank line between blocks, hyphen lists, concise paragraphs, and
  language-tagged code fences.
- Prefer generated discovery examples such as list_tasks() and list_models() over manually
  maintained exhaustive name lists.
- When changing models, package metadata, dependencies, public APIs, or runtime behavior, update
  the relevant README and this guidance if the workflow changes.
- For documentation-only updates, run git diff --check and verify relative links and headings.
