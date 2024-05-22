import torch
import torch.nn.functional as F
from collections import OrderedDict
from torchmeta.modules import MetaModule
import numpy as np


# def gradient_update_parameters(model,
#                                loss,
#                                params=None,
#                                step_size=0.5,
#                                first_order=False):
#     """Update of the meta-parameters with one step of gradient descent on the
#     loss function.
#
#     Parameters
#     ----------
#     model : `torchmeta.modules.MetaModule` instance
#         The model.
#
#     loss : `torch.Tensor` instance
#         The value of the inner-loss. This is the result of the training dataset
#         through the loss function.
#
#     params : `collections.OrderedDict` instance, optional
#         Dictionary containing the meta-parameters of the model. If `None`, then
#         the values stored in `model.meta_named_parameters()` are used. This is
#         useful for running multiple steps of gradient descent as the inner-loop.
#
#     step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
#         The step size in the gradient update. If an `OrderedDict`, then the
#         keys must match the keys in `params`.
#
#     first_order : bool (default: `False`)
#         If `True`, then the first order approximation of MAML is used.
#
#     Returns
#     -------
#     updated_params : `collections.OrderedDict` instance
#         Dictionary containing the updated meta-parameters of the model, with one
#         gradient update wrt. the inner-loss.
#     """
#     if not isinstance(model, MetaModule):
#         raise ValueError('The model must be an instance of `torchmeta.modules.'
#                          'MetaModule`, got `{0}`'.format(type(model)))
#
#     if params is None:
#         params = OrderedDict(model.meta_named_parameters())
#
#     grads = torch.autograd.grad(loss,
#                                 params.values(),
#                                 create_graph=not first_order)
#
#     updated_params = OrderedDict()
#
#     if isinstance(step_size, (dict, OrderedDict)):
#         for (name, param), grad in zip(params.items(), grads):
#             updated_params[name] = param - step_size[name] * grad
#
#     else:
#         for (name, param), grad in zip(params.items(), grads):
#             updated_params[name] = param - step_size * grad
#
#     return updated_params

def gradient_update_parameters(model,
                               loss,
                               train_input,
                               train_target,
                               params=None, velocity=None, momentum=None, t=1,
                               step_size=0.5,
                               first_order=False, adaptive=False, alpha=0.0005, sam_lower=True,
                               delta=0.001, beta1=0.9, beta2=0.99, 
                               isMomentum=True, beta=0.001):


    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())
    key_list = params.keys()
    items_list = params.values()
    params_list = list(params.values())

    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order)
    l = list(range(len(grads)))
    
    if momentum is None:
        momentum = []
        for i in l:
            momentum.append(torch.zeros_like(params_list[i]))
            
    if velocity is None:
        velocity = []
        for i in l:
            velocity.append(torch.zeros_like(params_list[i]) + 1e-12) 

    if sam_lower:
        gradnorm = grad_norm(params_list, grads, adaptive)
        # scale = alpha / (gradnorm + 1e-12)
        scale = (alpha / (gradnorm + 1e-12) - delta) 
        
        isMomentumTest = False

        old_p = []
        for i in l:
            old_p.append(torch.zeros_like(params_list[i]))

        for i in l:
            e_w = (torch.pow(params_list[i], 2) if adaptive else 1.0) * grads[i] * scale.to(params_list[i])
            params_list[i] = params_list[i].add(e_w)  # climb to the local maximum "w + e(w)"
                
                
                

        params_new = OrderedDict(zip(key_list, params_list))
        train_logit = model(train_input, params=params_new)
        inner_loss = F.cross_entropy(train_logit, train_target)
        model.zero_grad()
        grads_new = torch.autograd.grad(inner_loss,
                                        params_new.values(),
                                        create_graph=not first_order)      
                
        if isMomentum:
            for i in l:
                momentum[i] = beta1 * momentum[i] + (1 - beta1) * grads_new[i]
                velocity_temp = beta2 * velocity[i] + (1 - beta2) * torch.pow(grads_new[i], 2)
                #velocity_hat = torch.from_numpy(np.maximum(velocity[i].detach().numpy(), velocity_temp.detach().numpy()))
                velocity_hat = torch.from_numpy(np.maximum(velocity[i].cpu().numpy(), velocity_temp.cpu().numpy()))
                eta = torch.pow(torch.sqrt(velocity_hat), -1)
                eta = eta.to("cuda")
                momentum[i] = momentum[i].to("cuda")
                params_list[i] = params_list[i].to("cuda")
                velocity[i] = velocity_temp
                params_list[i] = params_list[i] - step_size * momentum[i] * eta

                params_mov = OrderedDict(zip(key_list, params_list))

                updated_params = OrderedDict()
                if isinstance(step_size, (dict, OrderedDict)):
                    for (name, param), grad in zip(params_mov.items(), grads_new):
                        updated_params[name] = param

                else:
                    for (name, param), grad in zip(params_mov.items(), grads_new):
                        updated_params[name] = param
                    
        else:
            updated_params = OrderedDict()
            if isinstance(step_size, (dict, OrderedDict)):
                for (name, param), grad in zip(params.items(), grads_new):
                    updated_params[name] = param - step_size[name] * grad

            else:
                for (name, param), grad in zip(params.items(), grads_new):
                    updated_params[name] = param - step_size * grad 

    else:
        updated_params = OrderedDict()
        if isinstance(step_size, (dict, OrderedDict)):
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - step_size[name] * grad

        else:
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - step_size * grad

    return updated_params, momentum, velocity

def grad_norm(params_list, grads, adaptive):
    shared_device = params_list[0].device  # put everything on the same device, in case of model parallelism
    l = list(range(len(grads)))
    norm = torch.norm(
                torch.stack([
                    ((torch.abs(params_list[i]) if adaptive else 1.0) * grads[i]).norm(p=2).to(shared_device)
                     for i in l
                    if grads is not None
                ]),
                p=2
           )
    return norm



