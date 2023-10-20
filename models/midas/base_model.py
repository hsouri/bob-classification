import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        #parameters = torch.load(path, map_location=torch.device('cpu'))
        parameters = torch.hub.load_state_dict_from_url(url=path, map_location="cpu", check_hash=True)

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
