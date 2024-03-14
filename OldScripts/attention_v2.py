from src.layers.rolling_attention import MLP_Rolling_attention2, RollingAttention2
from src.layers.model import Noise_Scheduler
from src.utils.training import Trainer
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from src.utils.datasets import get_dataset
from itertools import chain
import torch
import torch.nn as nn

torch.manual_seed(64)
# Variable intialisation
DEPTH = 4
SIZE = 256
HIDDEN_DIM = 256 * 4
OUTPUT_DIM = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 2

PATH = "Seq2Seq_LSTM.pth"


class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2), "Datasets must be the same length"
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        item1 = self.dataset1[idx]
        item2 = self.dataset2[idx]
        return item1, item2


dataset_moons = get_dataset("moons", 100000)
data_moons2 = get_dataset("moons", 100000)
combined_dataset = CombinedDataset(dataset_moons, data_moons2)

datset_circle = get_dataset("circle", 100000)
data_circle2 = get_dataset("circle", 100000)
combined_circles = CombinedDataset(datset_circle, data_circle2)

datset_line = get_dataset("dino", 100000)
datste_line2 = get_dataset("dino", 100000)
combined_lines = CombinedDataset(datset_line, datste_line2)

data_nine = get_dataset("mnist", n=100000, digit=9)
data_nine2 = get_dataset("mnist", n=100000, digit=9)
data_nine_comb = CombinedDataset(data_nine, data_nine2)

data_five = get_dataset("mnist", n=100000, digit=5)
data_five2 = get_dataset("mnist", n=100000, digit=5)
data_five_comb = CombinedDataset(data_five, data_five2)

concat = ConcatDataset([combined_lines, combined_dataset])
batch_size = 250
length_data = 100000 // 250
data_loader = DataLoader(concat, batch_size=batch_size, shuffle=False)

attention_model = RollingAttention2(
    input_dim=SIZE * 2, output_dim=SIZE, max_length=1000, device=DEVICE
)
Model = MLP_Rolling_attention2(
    depth=DEPTH,
    size=SIZE,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    attention_module=attention_model,
    device=DEVICE,
)
noise_scheduler_instance = Noise_Scheduler(beta_schedule="quadratic", device=DEVICE)

combined_params = chain(Model.parameters(), attention_model.parameters())

optimizer = torch.optim.Adam(combined_params, lr=0.0001)

for batch in data_loader:
    print(batch)
    break

trainer_instance = Trainer(
    Model,
    noise_scheduler_instance,
    optimizer,
    nn.MSELoss(),
    DEVICE,
    attention_model=attention_model,
    save_path=PATH,
)

print(len(data_loader))
losses, frames = trainer_instance.train_attention2(
    num_epochs=50,
    batch_size=250,
    gradient_clipthres=1.0,
    len_data=length_data,
    reset_probs=0.5,
    train_loader=data_loader,
)


print(losses)
print(frames)

# save this stuff

frames_np = np.stack(frames)
losses_np = np.array(losses)

np.save("framesconbined.npy", frames_np)
np.save("lossescombines.npy", losses_np)
