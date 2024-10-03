from abc import ABC
from typing import Any, Union, List, Tuple
import numpy as np
from deepface.commons import package_utils
from deepface.commons.logger import Logger

logger = Logger()

'''
tf_version = package_utils.get_tf_major_version()
if tf_version == 2:
    from tensorflow.keras.models import Model
else:
    from keras.models import Model
'''

# Notice that all facial recognition models must be inherited from this class

# pylint: disable=too-few-public-methods
class FacialRecognition(ABC):
    # model: Union[Model, Any]
    model_name: str
    input_shape: Tuple[int, int]
    output_shape: int

    def forward(self, img: np.ndarray) -> List[float]:
        logger.info("# FacialRecognition: using local model")
        # if not isinstance(self.model, Model):
        #     raise ValueError(
        #         "You must overwrite forward method if it is not a keras model,"
        #         f"but {self.model_name} not overwritten!"
        #     )
        # model.predict causes memory issue when it is called in a for loop
        # embedding = model.predict(img, verbose=0)[0].tolist()
        return []
        # return self.model(img, training=False).numpy()[0].tolist()
