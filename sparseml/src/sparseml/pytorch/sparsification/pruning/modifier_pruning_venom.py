
from typing import List, Union

from torch.nn import Module, Parameter, Linear
from torch.optim.optimizer import Optimizer

from sparseml.pytorch.sparsification.modifier import ModifierProp, PyTorchModifierYAML
from sparseml.pytorch.sparsification.pruning.mask_creator import (
    PruningMaskCreator,
    VNMPruningMaskCreator,
    get_mask_creator_default,
)
from sparseml.pytorch.sparsification.pruning.modifier_pruning_base import (
    BasePruningModifier,
)
from sparseml.pytorch.sparsification.pruning.modifier_pruning_magnitude import (
    MagnitudePruningParamsScorer,
)
from sparseml.pytorch.sparsification.pruning.scorer import PruningParamsScorer

from .venom_sparse_operations import (
    linear_to_venom_with_masks,
    linear_to_venom_detecting_mask
    
)

__all__ = ["VENOMKernelReplacementPruningModifier"]

@PyTorchModifierYAML()
class VENOMKernelReplacementPruningModifier(BasePruningModifier):
    """
    Pruning modifier that changes the model's linear modules to VENOM, 
    checking that the sparsity and mask type are valid.
    
    This pruning modifier only last one epoch, it changes the modules once at the start_epoch.
    
    """
    
    def __init__(
        self,
        at_epoch: Union[int, float],
        params: Union[str, List[str]],
        #global_sparsity: bool = True,
        leave_enabled: bool = True,
        mask_type: str = "unstructured",
        ):
        print("VENOMKernelReplacementPruningModifier initialization")
        if not (isinstance(mask_type, str) and mask_type.count(":") == 2):
            raise ValueError(
                "Unsupported mask type, expected a V:N:M mask, "
                f"got mask_type={mask_type}"
            )
        
        self._at_epoch = at_epoch
        self._mask_type = mask_type
        vnm = mask_type.split(":") 
        self._v=int(vnm[0])
        self._n=int(vnm[1])
        self._m=int(vnm[2])
        
        # Check N == 2, as it is th eonly value supported in the venom modules
        if self._n != 2:
            raise ValueError("Only masks with N = 2 are supported by VENOM modules. Got", self._n)
        
        # Calculate sparsity based on N and M parameters
        self._sparsity = float(self._n) / float(self._m)
                

        super(VENOMKernelReplacementPruningModifier, self).__init__(
            start_epoch=at_epoch,
            end_epoch=at_epoch, # This pruning modifier only last one epoch.
            #global_sparsity=global_sparsity,
            params=params,
            leave_enabled=leave_enabled,
        )
    
    
    
    def _get_mask_creator(
        self, param_names: List[str], params: List[Parameter]
    ) -> PruningMaskCreator:
        """
        :param names: full names of parameters to be pruned
        :param params: list of Parameters to be masked
        :return: mask creator object to be used by this pruning algorithm
        """
        self._mask_creator = VNMPruningMaskCreator(V=self._v, N=self._n, M=self._m)
        
        return self._mask_creator
        #return get_mask_creator_default(self.mask_type)
        
    def _get_scorer(self, params: List[Parameter]) -> PruningParamsScorer:
        """
        :param params: list of Parameters for scorer to track
        :return: param scorer object to be used by this pruning algorithm
        """
        return MagnitudePruningParamsScorer(params)
    
    def get_applied_sparsity_for_epoch(
        self, epoch: float, steps_per_epoch: int
    ) -> Union[float, List[float]]:
        """
        :param epoch: current epoch
        :param steps_per_epoch: number of steps per epoch
        :return: sparsity level that should be applied at the given epoch. If parameters
            should be set to different sparsities, should return a list of those values
            in the order the parameters appear in the mask manager for this object
        """
        return self._sparsity    
    
    def optimizer_pre_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        super().optimizer_pre_step(module, optimizer, epoch, steps_per_epoch)
        #print("Optimizer pre step on epoch", epoch)
        
    def optimizer_post_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        super().optimizer_post_step(module, optimizer, epoch, steps_per_epoch)
        # Replace model's linear modules with VENOM ones. 
        if epoch != self._at_epoch:
            #print("Not on at_epoch(", self._at_epoch, ") at the moment, currently at", epoch,"skipping...")
            return
        
        print("Optimizer post step on epoch", epoch)
        
        #print("Mask_creator mapping amount:\n", len(self._mask_creator.tensor_to_mask_mapping))
        #linear_to_venom_with_masks(module, self._v, self._n, self._m, self._mask_creator.tensor_to_mask_mapping)
        
        #for name, submod in module.named_children():
        #    if isinstance(submod, Linear):
        #        print(name, "tensor in dictionary?", submod.weight in self._mask_creator.tensor_to_mask_mapping)
        
        
        print("Replacing modules in VENOMKernelReplacementPruningModifier...")
        linear_to_venom_detecting_mask(module, self._v, self._n, self._m)
        print("Kernels replaced.")
    
    
    def is_oneshot(self, epoch: float, steps_per_epoch: int) -> bool:
        return True
    
    
    @ModifierProp()
    def mask_type(self) -> str:
        """
        :return: the mask type used
        """
        return self._mask_type
        
        