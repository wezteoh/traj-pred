from .relative_transformer import RelativeTransformer


def get_model(name, model_args, **kwargs):
    if name == "RelativeTransformer":
        return RelativeTransformer(**model_args)
    else:
        raise ValueError(f"Invalid model name: {name}")
