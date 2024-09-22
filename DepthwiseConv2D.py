from tensorflow.keras.layers import DepthwiseConv2D

# Definir una clase personalizada para DepthwiseConv2D
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Eliminar el argumento 'groups'
        super().__init__(*args, **kwargs)