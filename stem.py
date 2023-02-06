import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, data, batch_size=-1, device="cuda"):
    """
    from torchsummary import summary, change it for dict input
    """
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            if isinstance(input, (list, tuple)):
                # this is a sequential module for hook
                summary[m_key]["input_shape"] = list()
                # record input shape
                if isinstance(input[0], torch.Tensor):
                    input = input[0]
                else:
                    for l_i in input[0]:
                        summary[m_key]["input_shape"].append(l_i.size())
            if isinstance(input, torch.Tensor):
                summary[m_key]["input_shape"] = list(input.size())
            # the dict input wasn't a issues for me
            # if have some bugs, try fixed it.
            # if isinstance(input, dict):
            #     summary[m_key]["input_shape"] = input[0].size()

            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [batch_size] + list(o.size())[1:] for o in output
                ]
            elif isinstance(output, dict):
                summary[m_key]["output_shape"] = [k for k in output.keys()]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor


    # your need create your self input data before you call this function
    x = data
    input_size = []
    # get input shape
    if isinstance(x, torch.Tensor):
        input_size = data.size()
    if isinstance(x, (list, dict)):
        input_size = list(data.values())[0].size()
    if batch_size == -1:
        batch_size = input_size[0]
    input_size = input_size[1:]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # make a forward pass
    # my some net block need get the input shape then
    # to create the linear layer, so i need inject data before hook
    # print(x.shape)
    model(x)

    # some model need initialization after first forward
    # register hook
    model.apply(register_hook)

    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    print("--------------------------------------------------------------------------")
    line_new = "{:>25}  {:>30} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("==========================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params

        total_params += summary[layer]["nb_params"]
        # total_output += np.prod(summary[layer]["output_shape"])
        output_shape = summary[layer]["output_shape"]
        if isinstance(summary[layer]["output_shape"][0], list):
            output_shape = ""
            for out_shape_list in summary[layer]["output_shape"]:
                output_shape = f"{output_shape}  {out_shape_list}"
        if isinstance(summary[layer]['output_shape'][-1], int):
            total_output = summary[layer]['output_shape']
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]

        line_new = "{:>25}  {:>30} {:>15}".format(
            layer,
            str(output_shape),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * np.prod(total_output) * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("==========================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("--------------------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("--------------------------------------------------------------------------")
    # return summary
