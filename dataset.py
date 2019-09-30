from torch.utils.data import Dataset
from torchvision import datasets, transforms
from options import Options

class MyDataset(Dataset):
	def __init__(self, options:Options):
		transform = transforms.Compose([transforms.ToTensor()])

		self.data_train = datasets.MNIST(
			root = options.args.data_path,
			transform = transform,
			train = True,
			download = True)

	def __len__(self):
		return len(self.data_train)

	def __getitem__(self,index):
		return self.data_train[index]
