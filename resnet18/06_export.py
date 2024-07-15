import torchvision
import torch

BEST_MODEL_PATH = './model_best.pth'  # 最好的训练结果

def main(args=None):
  model = torchvision.models.resnet18(pretrained=False)
  model.fc = torch.nn.Linear(512,2)
  model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location='cpu'))
  device = torch.device('cpu')
  model = model.to(device)
  model.eval()
  x = torch.randn(1, 3, 224, 224, requires_grad=True)
  # torch_out = model(x)
  torch.onnx.export(model,
                    x,
                    BEST_MODEL_PATH[:-4] + ".onnx",
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'])

if __name__ == '__main__':
  main()