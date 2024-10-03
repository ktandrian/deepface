import numpy as np

# The type of float to use throughout a session.
_FLOATX = "float32"

# Default image data format, one of "channels_last", "channels_first".
_IMAGE_DATA_FORMAT = "channels_last"


def floatx():
    """Return the default float type, as a string.

    E.g. `'bfloat16'`, `'float16'`, `'float32'`, `'float64'`.

    Returns:
        String, the current default float type.

    Example:

    >>> keras.config.floatx()
    'float32'

    """
    return _FLOATX


def image_data_format():
    """Return the default image data format convention.

    Returns:
        A string, either `'channels_first'` or `'channels_last'`.

    Example:

    >>> keras.config.image_data_format()
    'channels_last'

    """
    return _IMAGE_DATA_FORMAT

def standardize_data_format(data_format):
    if data_format is None:
        return image_data_format()
    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    return data_format


def img_to_array(img, data_format=None, dtype=None):
    """Converts a PIL Image instance to a NumPy array.

    Example:

    ```python
    from PIL import Image
    img_data = np.random.random(size=(100, 100, 3))
    img = keras.utils.array_to_img(img_data)
    array = keras.utils.image.img_to_array(img)
    ```

    Args:
        img: Input PIL Image instance.
        data_format: Image data format, can be either `"channels_first"` or
            `"channels_last"`. Defaults to `None`, in which case the global
            setting `keras.backend.image_data_format()` is used (unless you
            changed it, it defaults to `"channels_last"`).
        dtype: Dtype to use. `None` means the global setting
            `keras.backend.floatx()` is used (unless you changed it, it
            defaults to `"float32"`).

    Returns:
        A 3D NumPy array.
    """

    data_format = standardize_data_format(data_format)
    if dtype is None:
        dtype = floatx()
    # NumPy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == "channels_first":
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == "channels_first":
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError(f"Unsupported image shape: {x.shape}")
    return x
