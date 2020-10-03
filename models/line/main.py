

import torch
from torch.utils.data import DataLoader
from utils import LineDataset
from line_model import Line
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pickle
from tqdm import tqdm

def main():
    iter_num = 3

    #input_file = 'data/edges3437_50remained_G1.pkl'
    #input_file = 'data/facebook_4k_50remained_G1.pkl'
    input_file = 'data/calls_500_50remained.pkl'
    #input_file = 'data/1k_fb_50remained.pkl'
    dataset = LineDataset(input_file)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate)

    model = Line(dataset.num_of_nodes, dim=64, order=2)

    optimizer = optim.Adam(model.parameters(), lr=0.025)
    batch_number = iter_num * len(dataloader)
    print('batch number: ', batch_number)
    scheduler = lr_scheduler.LambdaLR(optimizer,
                                           (lambda b: 1 - (b-1) / batch_number if 1 - (b-1) / batch_number > 0.0001 else 0.0001))

    for _ in tqdm(range(iter_num)):
        total_loss = 0
        for source, target, label in dataloader:
            label = torch.FloatTensor(label)

            optimizer.zero_grad()

            loss = model(source, target, label)
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.detach().item()

        print('\t Loss: %s' %(total_loss/len(dataloader)))

    #embed_dict = dataset.embedding_mapping(model.nodes_embed.data.numpy())

    #print('done converting nodes_embed ->  emb_dict')

    #with open('emb_dict.pickle', 'wb') as file:
    #    pickle.dump(embed_dict, file)

    model.save_embed(dataset.embedding_mapping, 'emb_dict.pickle')



if __name__ == '__main__':
    main()