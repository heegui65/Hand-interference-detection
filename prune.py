import torch
import torch_pruning as tp # type: ignore
from ultralytics import YOLO
from ruamel.yaml import YAML
import os

def prune_and_update_yaml(model_path, yaml_path, output_yaml, ratio=0.3):
    yolo = YOLO(model_path)
    model = yolo.model
    model.cpu()
    
    ignored_layers = []
    for m in model.modules():
        if hasattr(m, 'detect'): 
            ignored_layers.append(m)

    example_inputs = torch.randn(1, 3, 640, 640)
    imp = tp.importance.MagnitudeImportance(p=1) 
    
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=ratio,
        ignored_layers=ignored_layers,
    )
    
    pruner.step()
    
    def get_layer_out_channels(layer):

        for attr in ("cv2", "cv1", "conv", "dw", "downsample"):
            sub = getattr(layer, attr, None)
            if isinstance(sub, torch.nn.Conv2d):
                return sub.out_channels

            if hasattr(sub, 'conv') and isinstance(getattr(sub, 'conv'), torch.nn.Conv2d):
                return sub.conv.out_channels

        if isinstance(layer, torch.nn.Conv2d):
            return layer.out_channels
        convs = [m for m in layer.modules() if isinstance(m, torch.nn.Conv2d)]
        if convs:
            return convs[-1].out_channels

        return None

    pruned_channels = []
    for idx, layer in enumerate(getattr(model, 'model', [])):
        chan = get_layer_out_channels(layer)
        pruned_channels.append(chan)
        print(f"Layer{idx:2d} extracted out_channels: {chan}")


    yaml_tools = YAML()
    yaml_tools.preserve_quotes = True
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'gbk']
    config = None
    for enc in encodings:
        try:
            with open(yaml_path, 'r', encoding=enc) as f:
                config = yaml_tools.load(f)
            print(f" Loaded YAML with encoding: {enc}")
            break
        except Exception as e:
            print(f" YAML load with encoding {enc} failed: {e}")
            config = None
    if config is None:
        raise RuntimeError(f"Unable to read YAML file: {yaml_path}, please check encoding or if the file is corrupted")


    cfg_layers = []
    if 'backbone' in config and isinstance(config['backbone'], list):
        cfg_layers.extend(config['backbone'])
    if 'head' in config and isinstance(config['head'], list):
        cfg_layers.extend(config['head'])

    for i, layer_cfg in enumerate(cfg_layers):
        if i >= len(pruned_channels):
            print(f"Skipping cfg layer {i}: no corresponding pruned channel")
            continue
        chan = pruned_channels[i]
        if chan is None or not isinstance(chan, int) or chan <= 1:
            print(f"Skipping cfg layer {i} update: pruned channel not applicable ({chan})")
            continue
        try:
            if isinstance(layer_cfg, list) and len(layer_cfg) >= 4 and isinstance(layer_cfg[3], list) and len(layer_cfg[3]) > 0:
                layer_cfg[3][0] = int(chan)
                print(f"Updated cfg layer {i} channels -> {chan}")
        except Exception as e:
            print(f"Unable to update cfg layer {i}: {e}")

    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml_tools.dump(config, f)
    
    torch.save(model.state_dict(), "pruned_weights.pt")
    print(f"New YAML saved to: {output_yaml}")
    print(f"New weights saved to: pruned_weights.pt")

if __name__ == "__main__":
    prune_and_update_yaml(
        model_path=' ',
        yaml_path=' ',                 
        output_yaml=' ',         
        ratio=0.3                                     
    )