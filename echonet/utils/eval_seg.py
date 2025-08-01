"""Functions for training and running segmentation."""

import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

import echonet


@click.command("eval_seg")

@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)

@click.option("--weights_path", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.segmentation.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.segmentation.__dict__[name]))),
    default="deeplabv3_resnet50")


@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)

def run(
    data_dir=None,
    output=None,

    model_name="deeplabv3_resnet50",
    weights_path=None,

    save_video=False,
    
    num_workers=4,
    batch_size=20,
    device=None,
    seed=0,
):
    class EnsembleModel(torch.nn.Module):
        def __init__(self, models):
            super().__init__()
            self.models = torch.nn.ModuleList(models)

        def forward(self, x):
            outputs = [model(x) for model in self.models]
            out_stack = torch.stack([out['out'] for out in outputs], dim=0)

            ensemble_output = {}
            for key in outputs[0].keys():
                ensemble_output[key] = torch.stack([out[key] for out in outputs], dim=0).mean(dim=0)
            
            return ensemble_output
        
        def softmax_entropy(self, x):
            logit = torch.tensor(logit, dtype=torch.float32)

            logits = torch.stack([logit, -logit], dim=-1)

            probs = torch.softmax(logits, dim=-1)

            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

            return entropy.numpy() 
        
    def load_model(weights_path):
        model = torchvision.models.segmentation.__dict__[model_name](pretrained=False, aux_loss=False)
        model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)
        model = torch.nn.DataParallel(model)
        model.to(device)
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    def binary_softmax_entropy(logit):
        # Convert to tensor if it's not already
        logit = torch.tensor(logit, dtype=torch.float32)

        # Create binary logits: [logit, -logit]
        logits = torch.stack([logit, -logit], dim=-1)

        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1)

        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

        return entropy.numpy()  # Convert back to Python float

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = "./test"
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    weights = [f"{weights_path}/{i}/best.pt" for i in range(5)]
    print(weights)

    # === Load all models and build ensemble ===
    models = [load_model(path) for path in weights]
    model = EnsembleModel(models).to(device)
    model.eval()

    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std
              }

    # Set up datasets and dataloaders
    # for split in ["test"]:
    for split in ["val", "test"]:
        dataset = echonet.datasets.Echo(root=data_dir, split=split, **kwargs)
        print(f"DATASET LENGTH: {dataset.__len__}")
        dataloader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
        loss, large_inter, large_union, small_inter, small_union = echonet.utils.eval_seg.run_epoch(model, dataloader, False, None, device)

        overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
        large_dice = 2 * large_inter / (large_union + large_inter)
        small_dice = 2 * small_inter / (small_union + small_inter)
        with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
            g.write("Filename, Overall, Large, Small\n")
            for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                g.write("{},{},{},{}\n".format(filename, overall, large, small))

        print("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)), echonet.utils.dice_similarity_coefficient)))
        print("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(large_inter, large_union, echonet.utils.dice_similarity_coefficient)))
        print("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(small_inter, small_union, echonet.utils.dice_similarity_coefficient)))
        print()

   
    if save_video and not all(os.path.isfile(os.path.join(output, "videos", f)) for f in dataloader.dataset.fnames):
        dataset = echonet.datasets.Echo(root=data_dir, split="test",
                                            target_type=["Filename", "LargeIndex", "SmallIndex"],  # Need filename for saving, and human-selected frames to annotate
                                            mean=mean, std=std,  # Normalization
                                            length=None, max_length=None, period=1  # Take all frames
                                            )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=num_workers, shuffle=False, pin_memory=False, collate_fn=_video_collate_fn)


        model.eval()

        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        echonet.utils.latexify()

        with torch.no_grad():
            with open(os.path.join(output, "size.csv"), "w") as g:
                g.write("Filename,Frame,Size,HumanLarge,HumanSmall,ComputerSmall\n")
                for (x, (filenames, large_index, small_index), length) in tqdm.tqdm(dataloader):
                    # Run segmentation model on blocks of frames one-by-one
                    # The whole concatenated video may be too long to run together
                    y = np.concatenate([model(x[i:(i + batch_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], batch_size)])

                    start = 0
                    x = x.numpy()
                    for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                        # Extract one video and segmentation predictions
                        video = x[start:(start + offset), ...]
                        logit = y[start:(start + offset), 0, :, :]

                        print(logit.shape)
                        logit = binary_softmax_entropy(logit)

                        # Un-normalize video
                        video *= std.reshape(1, 3, 1, 1)
                        video += mean.reshape(1, 3, 1, 1)

                        # Get frames, channels, height, and width
                        f, c, h, w = video.shape  # pylint: disable=W0612
                        assert c == 3

                        # Put two copies of the video side by side
                        video = np.concatenate((video, video), 3)

                        # If a pixel is in the segmentation, saturate blue channel
                        # Leave alone otherwise
                        # video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])  # pylint: disable=E1111
                        video[:, 0, :, w:] = np.maximum(255. * logit, video[:, 0, :, w:])  # pylint: disable=E1111

                        # Add blank canvas under pair of videos
                        video = np.concatenate((video, np.zeros_like(video)), 2)

                        # Compute size of segmentation per frame
                        size = (logit > 0).sum((1, 2))

                        # Identify systole frames with peak detection
                        trim_min = sorted(size)[round(len(size) ** 0.05)]
                        trim_max = sorted(size)[round(len(size) ** 0.95)]
                        trim_range = trim_max - trim_min
                        systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])

                        # Write sizes and frames to file
                        for (frame, s) in enumerate(size):
                            g.write("{},{},{},{},{},{}\n".format(filename, frame, s, 1 if frame == large_index[i] else 0, 1 if frame == small_index[i] else 0, 1 if frame in systole else 0))

                        # Plot sizes
                        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                        ylim = plt.ylim()
                        for s in systole:
                            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                        plt.ylim(ylim)
                        plt.title(os.path.splitext(filename)[0])
                        plt.xlabel("Seconds")
                        plt.ylabel("Size (pixels)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf"))
                        plt.close(fig)

                        # Normalize size to [0, 1]
                        size -= size.min()
                        size = size / size.max()
                        size = 1 - size

                        # Iterate the frames in this video
                        for (f, s) in enumerate(size):

                            # On all frames, mark a pixel for the size of the frame
                            video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.

                            if f in systole:
                                # If frame is computer-selected systole, mark with a line
                                video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

                            def dash(start, stop, on=10, off=10):
                                buf = []
                                x = start
                                while x < stop:
                                    buf.extend(range(x, x + on))
                                    x += on
                                    x += off
                                buf = np.array(buf)
                                buf = buf[buf < stop]
                                return buf
                            d = dash(115, 224)

                            if f == large_index[i]:
                                # If frame is human-selected diastole, mark with green dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 225, 0]).reshape((1, 3, 1))
                            if f == small_index[i]:
                                # If frame is human-selected systole, mark with red dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 0, 225]).reshape((1, 3, 1))

                            # Get pixels for a circle centered on the pixel
                            r, c = skimage.draw.disk((int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))), 4.1)

                            # On the frame that's being shown, put a circle over the pixel
                            video[f, :, r, c] = 255.

                        # Rearrange dimensions and save
                        video = video.transpose(1, 0, 2, 3)
                        video = video.astype(np.uint8)
                        echonet.utils.savevideo(os.path.join(output, "videos", filename), video, 50)

                        # Move to next video
                        start += offset


def run_epoch(model, dataloader, train, optim, device):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """
    total = 0.
    n = 0

    pos = 0
    neg = 0
    pos_pix = 0
    neg_pix = 0

    model.train(train)

    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0
    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []

    # For misclassification detection
    all_entropy = []
    all_misclassified = []

    def binary_softmax_entropy(logit):
        # logit: (B, H, W)
        logits = torch.stack([logit, -logit], dim=-1)  # (B, H, W, 2)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # (B, H, W)
        return entropy

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (_, (large_frame, small_frame, large_trace, small_trace)) in dataloader:
                # Count number of pixels in/out of human segmentation
                pos += (large_trace == 1).sum().item()
                pos += (small_trace == 1).sum().item()
                neg += (large_trace == 0).sum().item()
                neg += (small_trace == 0).sum().item()

                pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
                pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
                neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
                neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()

                # === Large frames ===
                large_frame = large_frame.to(device)
                large_trace = large_trace.to(device)
                y_large = model(large_frame)["out"]
                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(y_large[:, 0, :, :], large_trace, reduction="sum")

                pred_large = (y_large[:, 0, :, :] > 0).float()
                correct_large = (pred_large == large_trace).float()
                entropy_large = binary_softmax_entropy(y_large[:, 0, :, :])

                all_entropy.append(entropy_large.detach().cpu().numpy().flatten())
                all_misclassified.append((1 - correct_large).detach().cpu().numpy().flatten())

                large_inter += np.logical_and(pred_large.detach().cpu().numpy() > 0., large_trace.detach().cpu().numpy() > 0.).sum()
                large_union += np.logical_or(pred_large.detach().cpu().numpy() > 0., large_trace.detach().cpu().numpy() > 0.).sum()
                large_inter_list.extend(np.logical_and(pred_large.detach().cpu().numpy() > 0., large_trace.detach().cpu().numpy() > 0.).sum((1, 2)))
                large_union_list.extend(np.logical_or(pred_large.detach().cpu().numpy() > 0., large_trace.detach().cpu().numpy() > 0.).sum((1, 2)))

                # === Small frames ===
                small_frame = small_frame.to(device)
                small_trace = small_trace.to(device)
                y_small = model(small_frame)["out"]
                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(y_small[:, 0, :, :], small_trace, reduction="sum")

                pred_small = (y_small[:, 0, :, :] > 0).float()
                correct_small = (pred_small == small_trace).float()
                entropy_small = binary_softmax_entropy(y_small[:, 0, :, :])

                all_entropy.append(entropy_small.detach().cpu().numpy().flatten())
                all_misclassified.append((1 - correct_small).detach().cpu().numpy().flatten())

                small_inter += np.logical_and(pred_small.detach().cpu().numpy() > 0., small_trace.detach().cpu().numpy() > 0.).sum()
                small_union += np.logical_or(pred_small.detach().cpu().numpy() > 0., small_trace.detach().cpu().numpy() > 0.).sum()
                small_inter_list.extend(np.logical_and(pred_small.detach().cpu().numpy() > 0., small_trace.detach().cpu().numpy() > 0.).sum((1, 2)))
                small_union_list.extend(np.logical_or(pred_small.detach().cpu().numpy() > 0., small_trace.detach().cpu().numpy() > 0.).sum((1, 2)))

                # Training step
                loss = (loss_large + loss_small) / 2
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # Logging
                total += loss.item()
                n += large_trace.size(0)
                p = pos / (pos + neg)
                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)
                pbar.set_postfix_str("{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}".format(
                    total / n / 112 / 112,
                    loss.item() / large_trace.size(0) / 112 / 112,
                    -p * math.log(p) - (1 - p) * math.log(1 - p),
                    (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(),
                    2 * large_inter / (large_union + large_inter),
                    2 * small_inter / (small_union + small_inter)
                ))
                pbar.update()

    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)

    # === Compute AUROC and AUPR ===
    all_entropy = np.concatenate(all_entropy)
    all_misclassified = np.concatenate(all_misclassified)

    auroc = roc_auc_score(all_misclassified, all_entropy)
    aupr = average_precision_score(all_misclassified, all_entropy)

    print(f"[Misclassification Detection] AUROC: {auroc:.4f}, AUPR: {aupr:.4f}")

    return (total / n / 112 / 112,
            large_inter_list,
            large_union_list,
            small_inter_list,
            small_union_list)


def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    #   height and width are expected to be the same across videos, but
    #   frames can be different.

    # ``target'' is also a tuple of length ``batch_size''
    # Each element is a tuple of the targets for the item.

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    # This contatenates the videos along the the frames dimension (basically
    # playing the videos one after another). The frames dimension is then
    # moved to be first.
    # Resulting shape is (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    target = zip(*target)

    return video, target, i
