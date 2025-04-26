# AugVid

**AugVid** is a collection of augmentation layers for videos inspired by the corresponding image preprocessing layers from `tf.keras`. 

<video src="https://github.com/user-attachments/assets/703b7934-4f37-4be3-8976-f38d41262bac"> demo </video>

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

## Demo

To generate demo video, first install the required dependencies:

```bash
pip install 'augvid[dev]'
```

Then run:

```bash
python demo.py --video <PATH_TO_VIDEO>
```
