import torch


class Tester:
    """
    Tester class
    """

    def __init__(self, test_data_loaders, models, device, metrics_epoch, test_metrics):
        self.test_data_loaders = test_data_loaders
        self.model = models["DQNet"]
        self.device = device
        self.metrics_epoch = metrics_epoch
        self.test_metrics = test_metrics

    def test(self):
        self.model.eval()
        with torch.no_grad():
            print("testing...")
            test_loader = self.test_data_loaders["MSU"]

            if len(self.metrics_epoch) > 0:
                outputs = torch.FloatTensor().to(self.device)
                targets = torch.FloatTensor().to(self.device)
            for batch_idx, (face, depth, target) in enumerate(test_loader):
                face, depth, target = face.to(self.device), depth.to(self.device), target.to(self.device).long()

                summap, output = self.model(face)
                output_depth = output.view(output.size(0), -1).mean(1)
                if len(self.metrics_epoch) > 0:
                    outputs = torch.cat((outputs, output_depth))
                    targets = torch.cat((targets, target))

                #
                # save sample images, or do something with output here
                #

            for met in self.metrics_epoch:
                self.test_metrics.epoch_update(met.__name__, met(targets, outputs))

        return targets, outputs, self.test_metrics.result()
