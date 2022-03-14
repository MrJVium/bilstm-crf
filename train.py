import json
import time
from os import mkdir
from processor import *
from utils import *
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

def __eval_model(model, device, dataloader, desc):
    model.eval()
    with torch.no_grad():
        losses, num = zip(*[
            (model.loss(x.to(device), y.to(device)), len(x))
            for x, y in tqdm(dataloader, desc=desc)
        ])
        return np.sum(np.multiply(losses, num)) / np.sum(num)

def __save_loss(losses, loss_path):
    pd.DataFrame(data=losses, columns=["epoch", "batch", "train_loss", "val_loss"]).to_csv(loss_path, index=False)

def __save_model(model_dir, model):
    model_path = model_filepath(model_dir)
    torch.save(model.state_dict(), model_path)
    print("save model => {}".format(model_path))

def train(args):
    model_dir = args.model_dir + '/' + str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    if not exists(model_dir):
        mkdir(model_dir)
    save_json_file(vars(args), arguments_filepath(model_dir))

    processor = Processor()
    model = build_model(args, processor, load=args.recovery, verbose=True)

    loss_path = join(args.model_dir, "loss.csv")
    losses = pd.read_csv(loss_path).values.tolist() if args.recovery and exists(loss_path) else []

    (train_xs, train_ys), (test_xs, test_ys), (valid_xs, valid_ys) = processor.load_dataset(args.val_split, args.test_split, args.max_seq_len)
    train_dl = DataLoader(TensorDataset(train_xs, train_ys), batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(TensorDataset(test_xs, test_ys), batch_size=args.batch_size * 2, shuffle=False)
    valid_dl = DataLoader(TensorDataset(valid_xs, valid_ys), batch_size=args.batch_size * 2, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_loss = 0
    best_val_loss = 1e4
    for epoch in range(args.num_epoch):
        model.train()
        bar = tqdm(train_dl)
        for bi, (xb, yb) in enumerate(bar):
            model.zero_grad()

            loss = model.loss(xb.to(device), yb.to(device))
            loss.backward()
            optimizer.step()
            bar.set_description("{:2d}/{} loss: {:5.2f}, val_loss: {:5.2f}".format(
                epoch, args.num_epoch, loss, val_loss))
            losses.append([epoch, bi, loss.item(), np.nan])

        val_loss = __eval_model(model, device, valid_dl, desc="eval")
        losses[-1][-1] = val_loss
        __save_loss(losses, loss_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            __save_model(args.model_dir, model)
            print("save model(epoch: {}) => {}".format(epoch, loss_path))

    # test
    test_loss = __eval_model(model, device, dataloader=test_dl, desc="test").item()
    last_loss = losses[-1][:]
    last_loss[-1] = test_loss
    losses.append(last_loss)
    __save_loss(losses, loss_path)
    print("training completed. test loss: {:.2f}".format(test_loss))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./seg-data/model")
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument('--max_seq_len', type=int, default=300)
    parser.add_argument('--num_rnn_layers', type=int, default=1)

    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--recovery', action="store_true")
    parser.add_argument('--save_best_val_model', action="store_true")

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()