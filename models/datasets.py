import torch


class MyDatasetSeq(torch.utils.data.Dataset):
    def __init__(self, data, action_reward_next_state=(None, None, None), transform=None):
        self.data = data
        self.transform = transform
        self.action, self.reward, self.next_state = action_reward_next_state

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            for idx, img in enumerate(x):
                transformed = self.transform(img)
                curr_img = torch.unsqueeze(transformed, 0)
                if idx == 0:
                    states = curr_img
                else:
                    states = torch.cat([states, curr_img], 0)

        items = states
        if self.action is not None:
            a = self.action[index]
            r = self.reward[index]

            z = self.next_state[index]
            for idx, img in enumerate(z):
                curr_img = torch.unsqueeze(self.transform(img), 0)
                if idx == 0:
                    next_states = curr_img
                else:
                    next_states = torch.cat([next_states, curr_img], 0)
            items = (states, a, r, next_states)

        return items

    def __len__(self):
        return len(self.data)
