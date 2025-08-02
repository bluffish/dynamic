import os
import click
import numpy as np
import torch
import torchvision
import tqdm
import matplotlib.pyplot as plt
import sklearn.metrics
import echonet
from scipy.stats import spearmanr

@click.command("eval_video")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default="./output_eval")
@click.option("--weights_path", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.video.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.video.__dict__[name]))),
    default="r2plus1d_18")
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=1)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
@click.option("--frames", type=int, default=32)
@click.option("--period", type=int, default=2)

def run(data_dir, output, weights_path, model_name, num_workers, batch_size, device, seed, frames, period):

    class HeteroscedasticEFModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            in_features = base_model.fc.in_features
            base_model.fc = torch.nn.Identity()
            self.base_model = base_model
            self.head = torch.nn.Linear(in_features, 2)  # outputs  [mean, log_var]

        def forward(self, x):
            features = self.base_model(x)
            out = self.head(features)
            mean = out[:, 0]
            log_var = out[:, 1]
            return mean, log_var
        

    class EnsembleModel(torch.nn.Module):
        def __init__(self, models):
            super().__init__()
            self.models = torch.nn.ModuleList(models)

        def forward(self, x):
            means, log_vars = [], []
            for model in self.models:
                mean, log_var = model(x)
                means.append(mean)
                log_vars.append(log_var)
            means = torch.stack(means, dim=0)  # (ensemble, batch)
            log_vars = torch.stack(log_vars, dim=0)
            return means.mean(0), means.var(0), log_vars.mean(0)  # pred, epistemic, aleatoric


    def load_model(path):
        base = torchvision.models.video.__dict__[model_name](pretrained=False)
        model = HeteroscedasticEFModel(base)
        model = torch.nn.DataParallel(model).to(device)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    np.random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output, exist_ok=True)

    weight_paths = [os.path.join(weights_path, f"{i}/best.pt") for i in range(5)]
    models = [load_model(p) for p in weight_paths]
    model = EnsembleModel(models).to(device)

    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))

    kwargs = {
        "target_type": "EF",
        "mean": mean,
        "std": std,
        "length": frames,
        "period": period,
        "clips": "all"
    }

    for split in ["val", "test"]:
        dataset = echonet.datasets.Echo(root=data_dir, split=split, **kwargs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))

        all_preds = []
        all_vars = []
        all_targets = []
        al_vars = []
        ep_vars = []

        with torch.no_grad():
            with tqdm.tqdm(total=len(dataloader)) as pbar:
                for x, y in dataloader:
                    x = x.to(device)  # (1, clips, C, F, H, W)
                    # print(x.shape)

                    if kwargs['clips'] == 'all':
                        b, c, f, h, w = x.shape[1:]
                    else:
                        b, c, f, h, w = x.shape

                    x = x.view(-1, c, f, h, w)

                    yhat, epistemic_var, aleatoric_logvar = model(x)
                    yhat = yhat.view(-1).cpu().numpy()
                    var = var.view(-1).cpu().numpy()

                    al_vars.append(torch.exp(aleatoric_logvar).view(-1).cpu().numpy())
                    ep_vars.append(epistemic_var.view(-1).cpu().numpy())

                    all_preds.append(yhat)
                    all_vars.append(var)
                    all_targets.append(y.numpy()[0])
                    pbar.update()

        yhat_mean = np.array([pred.mean() for pred in all_preds])
        y = np.array(all_targets)

        r2 = sklearn.metrics.r2_score(y, yhat_mean)
        mae = sklearn.metrics.mean_absolute_error(y, yhat_mean)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y, yhat_mean))

        print(f"Variance: {np.array(all_vars).mean()}")

        print(f"{split} R2: {r2:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        with open(os.path.join(output, f"{split}_predictions.csv"), "w") as f:
            for fname, preds in zip(dataset.fnames, all_preds):
                for i, p in enumerate(preds):
                    f.write(f"{fname},{i},{p:.4f}\n")

        fig = plt.figure(figsize=(3, 3))
        plt.scatter(y, yhat_mean, color="k", s=1)
        plt.plot([0, 100], [0, 100], linewidth=1, linestyle="--")
        plt.xlabel("Actual EF (%)")
        plt.ylabel("Predicted EF (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output, f"{split}_scatter.pdf"))
        plt.close(fig)

        corr, _ = spearmanr(abs_errors, epistemic_vars)
        print(f"Spearman correlation between |error| and epistemic uncertainty: {corr:.3f}")
