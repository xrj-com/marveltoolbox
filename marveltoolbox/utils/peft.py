import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import copy
    

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.base = base_layer  
        self.base.weight.requires_grad = False
        self.base.bias.requires_grad = False
        self.r = r
        self.alpha = alpha

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.scaling = alpha / r
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 0

    def forward(self, x):
        if self.r > 0:
            delta_w = torch.matmul(self.lora_B, self.lora_A)  # [out, in]
            result= nn.functional.linear(x, self.scaling * delta_w + self.base.weight, self.base.bias)
        else:
            result= nn.functional.linear(x, self.base.weight, self.base.bias)
        return result
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"r={self.r}, alpha={self.alpha})")
    
    

class LoRAConv2d(nn.Module):
    def __init__(self, base_layer: nn.Conv2d, r=4, alpha=1.0):
        super().__init__()
        self.base = base_layer
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.r = r
        self.alpha = alpha

        self.in_channels = base_layer.in_channels
        self.out_channels = base_layer.out_channels
        self.kernel_size = base_layer.kernel_size
        self.stride = base_layer.stride
        self.padding = base_layer.padding
        self.dilation = base_layer.dilation
        self.groups = base_layer.groups

        if self.r > 0:
            # A: [r, in_channels * kH * kW]
            self.lora_A = nn.Parameter(
                torch.empty(
                    self.r, 
                    (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1]
                )
            )
            # B: [out_channels, r]
            self.lora_B = nn.Parameter(torch.empty(self.out_channels, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.scaling = self.alpha / self.r if self.r > 0 else 1.0

    def forward(self, x):
        if self.r > 0:
            lora_weight_2d = torch.mm(self.lora_B, self.lora_A)

            merged_weight_2d = lora_weight_2d.view(
                self.out_channels, 
                self.in_channels // self.groups, 
                self.kernel_size[0], 
                self.kernel_size[1]
            ) * self.scaling + self.base.weight

            out = F.conv2d(
                x, 
                merged_weight_2d, 
                self.base.bias, 
                stride=self.stride, 
                padding=self.padding, 
                dilation=self.dilation, 
                groups=self.groups
            )

        else:
            out = self.base(x)

        return out
    
    def __repr__(self):
        return (f"LoRAConv2d({self.base.in_channels}, {self.base.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
                f"dilation={self.dilation}, groups={self.groups}, r={self.r}, alpha={self.alpha})")
    

def extract_lora_state_dict(model):
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            lora_state[f"{name}.lora_A"] = module.lora_A
            lora_state[f"{name}.lora_B"] = module.lora_B
    return lora_state

def get_lora_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]

def inject_lora(model: nn.Module,
                target_replace_modules=(nn.Linear, nn.Conv2d),
                r: int = 4,
                alpha: float = 1.0,
                lora_state_dict: dict | None = None,
                strict: bool = True) -> nn.Module:
    """Inject LoRA into a model."""
    device = next(model.parameters()).device
    injected = copy.deepcopy(model)             
    _inject_inplace(injected,
                    target_replace_modules,
                    r, alpha,
                    lora_state_dict,
                    prefix="",
                    strict=strict)              
    return injected.to(device)

def _inject_inplace(module: nn.Module,
                    target_replace_modules,
                    r: int,
                    alpha: float,
                    lora_state_dict: dict | None,
                    prefix: str,
                    strict: bool):
    """Inject LoRA into a model inplace."""
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # -------- nn.Linear --------
        if isinstance(child, nn.Linear) and nn.Linear in target_replace_modules:
            new_child = LoRALinear(child, r=r, alpha=alpha)
            if lora_state_dict:
                _load_lora_weights(new_child, lora_state_dict, full_name, strict)
            module._modules[name] = new_child

        # -------- nn.Conv2d --------
        elif isinstance(child, nn.Conv2d) and nn.Conv2d in target_replace_modules:
            new_child = LoRAConv2d(child, r=r, alpha=alpha)
            if lora_state_dict:
                _load_lora_weights(new_child, lora_state_dict, full_name, strict)
            module._modules[name] = new_child
        else:
            _inject_inplace(child,
                            target_replace_modules,
                            r, alpha,
                            lora_state_dict,
                            prefix=full_name,
                            strict=strict)


def _load_lora_weights(lora_module, state_dict, full_name, strict):
    """Load LoRA weights from state_dict into lora_module."""
    try:
        lora_module.lora_A.data.copy_(state_dict[f"{full_name}.lora_A"])
        lora_module.lora_B.data.copy_(state_dict[f"{full_name}.lora_B"])
    except KeyError as e:
        if strict:
            raise RuntimeError(f"Missing LoRA weights for {full_name}: {e}")

def merge_lora_weights(model: nn.Module) -> nn.Module:
    device = next(model.parameters()).device
    merged = copy.deepcopy(model)        
    _merge_inplace(merged)               
    return merged.to(device)

def _merge_inplace(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            delta_W = (child.lora_B @ child.lora_A) * child.scaling
            child.base.weight.data += delta_W
            setattr(module, name, child.base)      

        elif isinstance(child, LoRAConv2d):
            delta_W = (child.lora_B @ child.lora_A).view_as(
                child.base.weight
            ) * child.scaling
            child.base.weight.data += delta_W
            setattr(module, name, child.base)

        else:
            _merge_inplace(child)

if __name__ == "__main__":
    import torch
    import torch.nn as nn

    model = torch.nn.Sequential(
        nn.Linear(50,100),
        nn.ReLU(),
        nn.Linear(100,10),
    )
    model_lora = inject_lora(model, r=4, alpha=1.0)
    merged_model = merge_lora_weights(model_lora)

    print(model)
    print(model_lora)
    print(merged_model)

    parameters = extract_lora_state_dict(model_lora)
    print(parameters)
    model_lora_2 = inject_lora(model, r=4, alpha=1.0, lora_state_dict=parameters)
    print(model_lora_2)
    # # 随机输入数据
    # input_data = torch.randn(1, 3, 32, 32)      

    # # 前向传播
    # base_out = conv_layer(input_data)
    # lora_out = lora_conv_layer(input_data)
    # print("Base Conv Layer Output:", base_out.shape)
    # print("LoRA Conv Layer Output:", lora_out.shape)
    # assert torch.allclose(base_out, lora_out), "The outputs of the base and LoRA layers should be close."
    # print("LoRA layer is working correctly!")

    # # 检查LoRA参数数量
    # lora_params = sum(p.numel() for p in lora_conv_layer.parameters())
    # print(f"Number of LoRA parameters: {lora_params}")
    # assert lora_params < 1000, "The number of LoRA parameters should be less than 1000."
    # print("LoRA layer has a reasonable number of parameters!")
