from tensorflow.python.client import device_lib


def available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [device.name for device in local_device_protos if device.device_type == 'GPU']
