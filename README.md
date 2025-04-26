# AugVid

**AugVid** is a collection of augmentation layers for videos inspired by the corresponding image preprocessing layers from `tf.keras`. 

<video loop src="https://github.com/user-attachments/assets/703b7934-4f37-4be3-8976-f38d41262bac"> video </video>

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

