import argparse

import torch
from torch.utils.data import DataLoader

from model import ModelClass
from utils import RecommendationDataset

from torch import nn
from math import sqrt
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2021 AI Final Project')
    parser.add_argument('--save-model', default='model.pt', help="Model's state_dict")
    parser.add_argument('--dataset', default='./data', help='dataset directory')
    parser.add_argument('--batch-size', default=16, help='train loader batch size')

    args = parser.parse_args()

    # instantiate model
    model = ModelClass()
    
    # load dataset in train folder
    train_data = RecommendationDataset(f"{args.dataset}/ratings.csv", train=True)
    valid_data = RecommendationDataset(f"{args.dataset}/ratings.csv", train=True)
    
    train_data.data_pd, valid_data.data_pd = train_test_split(train_data.data_pd, test_size=0.1, shuffle=True, random_state=34)
    
    train_data.items = torch.LongTensor(train_data.data_pd['itemId'])
    train_data.users = torch.LongTensor(train_data.data_pd['userId'])
    train_data.ratings = torch.FloatTensor(train_data.data_pd['rating'])
    
    valid_data.items = torch.LongTensor(valid_data.data_pd['itemId'].values)
    valid_data.users = torch.LongTensor(valid_data.data_pd['userId'].values)
    valid_data.ratings = torch.LongTensor(valid_data.data_pd['rating'].values)

    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    
    _, _, n_ratings = train_data.get_datasize()
    _, _, val_n_ratings = valid_data.get_datasize()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay = 1e-5)
    criterion = nn.MSELoss()
 
    
    for epoch in range(8):
        cost = 0

        for users, items, ratings in train_loader:
            optimizer.zero_grad()
            ratings_pred = model(users, items)
            loss = torch.sqrt(criterion(ratings_pred, ratings))
            loss.backward()
            optimizer.step()
            cost += loss.item() * len(ratings)
               
        cost /= n_ratings
      
        print(f"Epoch : {epoch}")
        print("train cost : {:.6f}".format(cost))
        
        cost_valid = 0
        
        for users, items, ratings in valid_loader:
            ratings_pred = model(users, items)
            loss = torch.sqrt(criterion(ratings_pred, ratings))
            cost_valid += loss.item() * len(ratings)
        
        cost_valid /= val_n_ratings
       
        
        print("valid cost : {:.6f}".format(cost_valid))
                             

    torch.save(model.state_dict(), args.save_model)