# %%
import resnetv2
import torch
import torchvision as tv
import numpy as np
from utils.test_utils import arg_parser
import os
import pickle

# %%
def make_id_ood(args, out_dataset):
    """Returns train and validation datasets."""
    crop = 480

    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
    ])

    in_set = tv.datasets.ImageFolder(args.in_datadir, val_tx)
    out_set = tv.datasets.ImageFolder(os.path.join(args.out_datadir,out_dataset), val_tx)

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return in_set, out_set, in_loader, out_loader

def iterate_data_feature_score(data_loader, model):
    m = torch.nn.Softmax(dim=-1).cuda()
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    _MaxLogit = []
    _MSP = []
    _ODIN = []
    _energy = []
    _xent = []
    _feature = []
    normalizer = tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    for b, (x, y) in enumerate(data_loader):
        print("dealing with batch {}".format(b))
        with torch.no_grad():
            feature, outputs = model(normalizer(x.cuda()), return_feature = True)
            _feature.append(feature.cpu().detach().numpy())
            outputs = outputs 
            
            MaxLogit = torch.max(outputs, dim=-1)[0]
            _MaxLogit.extend(MaxLogit.data.cpu().numpy())

            MSP = torch.max(m(outputs), dim=-1)[0]
            _MSP.extend(MSP.data.cpu().numpy())
            
            temperature = args.temperature_odin
            ODIN = torch.max(m(outputs / temperature), dim=-1)[0]
            _ODIN.extend(ODIN.data.cpu().numpy())
            
            temperature = args.temperature_energy
            energy = temperature * torch.logsumexp(outputs, dim=1)
            _energy.extend(energy.data.cpu().numpy())
            
            targets = torch.ones_like(outputs)/outputs.shape[1]
            xent = torch.sum(-targets * logsoftmax(outputs), dim=-1)
            _xent.extend(xent.data.cpu().numpy())
            

    with open("./save_feature/{}_features".format(data_loader.name), "wb") as f:
        pickle.dump(np.concatenate(_feature), f)
    
    
    with open("./save_feature/{}_xent".format(data_loader.name), "wb") as f:
        pickle.dump(np.array(_xent), f)
    
    with open("./save_feature/{}_energy".format(data_loader.name), "wb") as f:
        pickle.dump(np.array(_energy), f)
    
    with open("./save_feature/{}_MSP".format(data_loader.name), "wb") as f:
        pickle.dump(np.array(_MSP), f)
    
    with open("./save_feature/{}_ODIN".format(data_loader.name), "wb") as f:
        pickle.dump(np.array(_ODIN), f)
    
    print("Successfully preprocessed!")
    return

# %%
parser = arg_parser()

parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")

# arguments for ODIN
parser.add_argument('--temperature_odin', default=100, type=int,
                    help='temperature scaling for odin')
parser.add_argument('--epsilon_odin', default=0.0, type=float,
                    help='perturbation magnitude for odin')
# arguments for Energy
parser.add_argument('--temperature_energy', default=1, type=int,
                    help='temperature scaling for energy')

# %%
args = parser.parse_args()

# %%
os.makedirs(f"./save_feature", exist_ok = True)
in_set, out_set, in_loader, out_loader = make_id_ood(args, out_dataset = "iNaturalist")

torch.backends.cudnn.benchmark = True
model = resnetv2.KNOWN_MODELS[args.model](head_size=len(in_set.classes))
state_dict = torch.load(args.model_path)
model.load_state_dict_custom(state_dict['model'])
model = model.cuda()
model.eval()

# %%
in_loader.name = "imagenet_val"
out_loader.name = "iNaturalist"
iterate_data_feature_score(in_loader, model)
iterate_data_feature_score(out_loader, model)

# %%
in_set, out_set, in_loader, out_loader = make_id_ood(args, "Places")
out_loader.name = "Places"
iterate_data_feature_score(out_loader, model)

# %%
in_set, out_set, in_loader, out_loader = make_id_ood(args, "dtd/images")
out_loader.name = "Textures"
iterate_data_feature_score(out_loader, model)

# %%
in_set, out_set, in_loader, out_loader = make_id_ood(args, "SUN")
out_loader.name = "SUN"
iterate_data_feature_score(out_loader, model)