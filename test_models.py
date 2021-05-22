import numpy as np
import torch
import torchvision

from os import listdir
from sklearn.model_selection import train_test_split

from models.autoencoders import EncoderSmall, DecoderSmall, CnnAE
from models.autoencoders import DeepEncoderSmall, DeepDecoderSmall
from models.autoencoders import SmallEncoder40, SmallDecoder40

files_here = listdir()
np_files = [x for x in files_here if x.endswith('.npy')]
actions = np.array([])
next_states_scaled = np.array([])
scaled = np.array([])

for idx, file_name in enumerate(np_files):
  with open(file_name, 'rb') as f:
    current_array = np.load(f) 
  print(file_name)
  if file_name.startswith('states_scaled'):
    if len(scaled) == 0:
      scaled = current_array
    else:
      scaled = np.concatenate([scaled, current_array])
  elif file_name.startswith('next_states_scaled'):
    if len(next_states_scaled) == 0:
      next_states_scaled = current_array
    else:
      next_states_scaled = np.concatenate([next_states_scaled, current_array])
  elif file_name.startswith('actions'):
    if len(actions) == 0:
      actions = current_array
    else:
      actions = np.concatenate([actions, current_array])
train_scaled, eval_scaled, train_next_states, eval_next_states = train_test_split(
    scaled, next_states_scaled, test_size=0.15, random_state=42)

train_actions, eval_actions = train_test_split(actions, test_size=0.15, random_state=42)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, action_next_state=(None, None), transform=None):
        self.data = data
        self.transform = transform
        self.action, self.next_state = action_next_state

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        items = x
        if self.action is not None:
            y = self.action[index]
            z = self.next_state[index]
            z = self.transform(z)
            items = (x, y, z)

        return items

    def __len__(self):
        return len(self.data)

means = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
stds = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
to_tensor = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                                        mean=means,
                                                        std=stds)
                                             ])


scaled_states_actions_dataset = MyDataset(train_scaled, (train_actions, train_next_states), transform=to_tensor)
scaled_states_actions_loader = torch.utils.data.DataLoader(scaled_states_actions_dataset, batch_size=64,
                                                    shuffle=True)

eval_scaled_states_actions_dataset = MyDataset(eval_scaled, (eval_actions , eval_next_states), transform=to_tensor)
eval_scaled_states_actions_loader = torch.utils.data.DataLoader(eval_scaled_states_actions_dataset, batch_size=2,
                                                    shuffle=True)

_, (state, action, next_state) = next(enumerate(scaled_states_actions_loader))

encoder = EncoderSmall(9)
decoder = DecoderSmall(9)
ae = CnnAE(encoder, decoder)


print(encoder)
print(decoder)
print(ae)



encoder = DeepEncoderSmall(9)
decoder = DeepDecoderSmall(9)
ae = CnnAE(encoder, decoder)


print(encoder)
print(decoder)
print(ae)


encoder = SmallEncoder40 (9, [32, 64, 128], non_variational=True)
decoder = SmallDecoder40(9,  list(reversed([32, 64, 128])))
ae = CnnAE(encoder, decoder)
encoded = encoder(state, debug=True)
decoded = decoder(encoded, debug=True)
