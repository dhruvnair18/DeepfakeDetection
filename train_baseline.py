import os, argparse, itertools
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def loaders(root, img=224, bs=32):
    ttrain = transforms.Compose([
        transforms.RandomResizedCrop(img, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    tval = transforms.Compose([
        transforms.Resize(img),
        transforms.CenterCrop(img),
        transforms.ToTensor()
    ])
    ds_tr = datasets.ImageFolder(Path(root)/"train", transform=ttrain)
    ds_va = datasets.ImageFolder(Path(root)/"val",   transform=tval)
    ld_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True,  num_workers=2)
    ld_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=2)
    return ld_tr, ld_va, ds_tr.classes

def train_one(model, loader, crit, opt, dev):
    model.train(); tl=0; acc=0; n=0
    for x,y in loader:
        x,y = x.to(dev), y.to(dev)
        opt.zero_grad(); out=model(x); loss=crit(out,y); loss.backward(); opt.step()
        tl += loss.item()*x.size(0); acc += (out.argmax(1)==y).sum().item(); n += x.size(0)
    return tl/n, acc/n

@torch.no_grad()
def eval_one(model, loader, crit, dev):
    model.eval(); tl=0; acc=0; n=0; Y=[]; P=[]
    for x,y in loader:
        x,y = x.to(dev), y.to(dev)
        out=model(x); loss=crit(out,y)
        tl += loss.item()*x.size(0); acc += (out.argmax(1)==y).sum().item(); n += x.size(0)
        Y += y.cpu().tolist(); P += out.argmax(1).cpu().tolist()
    return tl/n, acc/n, Y, P

def plot_cm(y, p, classes, path):
    cm = confusion_matrix(y,p)
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix"); plt.colorbar()
    ticks = range(len(classes)); plt.xticks(ticks, classes); plt.yticks(ticks, classes)
    thr = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j],ha="center", color="white" if cm[i,j]>thr else "black")
    plt.ylabel("True"); plt.xlabel("Predicted"); Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args=ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)

    tr, va, classes = loaders(args.data, bs=args.bs)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(dev)

    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=args.lr)

    best=0.0
    for e in range(1, args.epochs+1):
        tl, ta = train_one(model, tr, crit, opt, dev)
        vl, va_acc, Y, P = eval_one(model, va, crit, dev)
        print(f"Epoch {e}: train_acc={ta:.3f}  val_acc={va_acc:.3f}")
        if va_acc>best:
            best=va_acc; os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/mobilenet_v2_baseline.pt")

    print(classification_report(Y, P, target_names=classes, digits=3))
    plot_cm(Y, P, classes, "reports/confusion_matrix.png")
    print("Saved: checkpoints/mobilenet_v2_baseline.pt  and  reports/confusion_matrix.png")
