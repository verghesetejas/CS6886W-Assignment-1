import argparse, torch, torch.nn as nn
from models.vgg6 import VGG6
from data.cifar10 import loaders
from training.engine import epoch

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",default="outputs/best.pt"); a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); c=ck["config"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _,_,test=loaders(c["data_dir"],c["batch_size"],c["num_workers"],c["seed"],c["cutout_size"])
    model=VGG6(act=c["activation"]).to(device); model.load_state_dict(ck["model_state_dict"])
    loss,acc=epoch(model,test,nn.CrossEntropyLoss(),device,amp=c["use_amp"])
    print(f"Test loss: {loss:.4f}; test top-1 accuracy: {acc:.3f}%")
if __name__=="__main__": main()
