#!/usr/bin/env python3

"""
MAML module as seen in https://github.com/learnables/learn2learn
with the addition of the functions "get_rep" and "get_rep_i"
"""

from learn2learn.algorithms.maml import MAML as MAML_BASE
from learn2learn.utils import clone_module


class MAML(MAML_BASE):
    pass

    def get_rep(self, input_d):
        return self.get_base_representation(input_d)

    def get_rep_i(self, input_d, layer_i):
        return self.get_rep_layer(input_d, layer_i)

    # After adding the new above functions the clone module also needs
    # to be reloaded in order to copy the above functions as well
    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Description**
        Returns a `MAML`-wrapped copy of the module whose parameters and buffers
        are `torch.clone`d from the original module.
        This implies that back-propagating losses on the cloned module will
        populate the buffers of the original module.
        For more information, refer to learn2learn.clone_module().
        **Arguments**
        * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
            or second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAML(clone_module(self.module),
                    lr=self.lr,
                    first_order=first_order,
                    allow_unused=allow_unused,
                    allow_nograd=allow_nograd)
