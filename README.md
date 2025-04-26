# AugVid

<video width="640" controls>
  <source src="assets/demo.mp4" type="video/mp4">
</video>

<video loop src="assets/demo.mp4"> video </video>

**AugVid** is a collection of augmentation layers for videos inspired by the corresponding image preprocessing layers from `tf.keras`. 

## Installation

```bash
pip install augvid
```

## Getting Started

The augmentation layers can be added during the model construction:

```python
import keras
from augvid import RandomVideoBrightness, RandomHorizontalVideoFlip


model = keras.Sequential([
    RandomVideoBrightness(max_delta=0.1),
    RandomHorizontalVideoFlip(),
    # add more layers here
])
```

