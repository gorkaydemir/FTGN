import zlib
import pickle
from torch.utils.data import Dataset


class Argoverse_Edge_Dataset(Dataset):
    def __init__(self, args, validation=False):

        self.ex_file_path = args.ex_file_path
        if validation:
            self.ex_file_path = args.val_ex_file_path
        if args.test:
            self.ex_file_path = args.test_ex_file_path

        self.device = args.device

        pickle_file = open(self.ex_file_path, 'rb')
        self.ex_list = pickle.load(pickle_file)
        pickle_file.close()

    def __len__(self):
        return len(self.ex_list)

    def get_mapping(self, idx):
        data_compress = self.ex_list[idx]
        instance = pickle.loads(zlib.decompress(data_compress))
        return instance

    def __getitem__(self, idx):
        mapping = self.get_mapping(idx)
        return mapping
