from captum.attr import IntegratedGradients


def integrated_gradients(model, input_tensor, target=1, n_steps=32):
    ig = IntegratedGradients(model)
    return ig.attribute(input_tensor, target=target, n_steps=n_steps)