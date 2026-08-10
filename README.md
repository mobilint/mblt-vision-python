# mblt-vision-python

**Vision models** for Mobilint NPUs.

Image classification, object detection, instance segmentation and pose estimation on
Mobilint NPUs. Each model is a class that carries its own pre- and post-processing:

```python
from mblt_vision import ResNet50

model = ResNet50()
x = model.preprocess("image.jpg")
result = model.postprocess(model(x))
```

Split out of `mblt-model-zoo`, which still re-exports it as `mblt_model_zoo.vision`
so existing code keeps working.

## Installation

```bash
pip install mblt-vision-python
```

## License

BSD-3-Clause.
