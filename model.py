from torch.autograd import Variable
import torch.nn as nn
from Networks import VGG_Network, InceptionV3_Network, Resnet50_Network
from torch import optim
import torch
import time
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cal_dis(f_p,f_s):
    
    dis=f_p-f_s
    dis=torch.square(dis)
    dis=torch.sum(dis,dim=1)
    D=torch.sum(dis,dim=[1,2])
    D=torch.sqrt(D)

    return(D)
class FGSBIR_Model(nn.Module):
    def __init__(self, hp):
        super(FGSBIR_Model, self).__init__()
        self.sample_embedding_network_s = eval(hp.backbone_name + '_Network(hp)')
        self.sample_embedding_network_p=eval(hp.backbone_name + '_Network(hp)')
        self.loss = nn.TripletMarginLoss(margin=0.2)
        self.sample_train_params_s = self.sample_embedding_network_s.parameters()
        self.sample_train_params_p=self.sample_embedding_network_p.parameters()
        self.optimizer_p = optim.Adam(self.sample_train_params_p, hp.learning_rate)
        self.optimizer_s=optim.Adam(self.sample_train_params_s,hp.learning_rate)
        self.hp = hp


    def train_model(self, batch):
        self.train()
        self.optimizer_p.zero_grad()  
        self.optimizer_s.zero_grad()
        positive_feature = self.sample_embedding_network_p(batch['positive_img'].to(device))
        negative_feature = self.sample_embedding_network_p(batch['negative_img'].to(device))
        sample_feature = self.sample_embedding_network_s(batch['sketch_img'].to(device))
        
        positive_dis=cal_dis(positive_feature,sample_feature)
        negative_dis=cal_dis(negative_feature,sample_feature)
        loss=F.relu(positive_dis-negative_dis+0.2).mean()
        #loss = self.loss(sample_feature, positive_feature, negative_feature)
        loss.backward()
        self.optimizer_p.step()
        self.optimizer_s.step()

        return loss.item() 

    def evaluate(self, datloader_Test):
        Image_Feature_ALL = []
        Image_Name = []
        Sketch_Feature_ALL = []
        Sketch_Name = []
        start_time = time.time()
        self.eval()
        for i_batch, sanpled_batch in enumerate(datloader_Test):
            sketch_feature, positive_feature= self.test_forward(sanpled_batch)
            Sketch_Feature_ALL.extend(sketch_feature)
            Sketch_Name.extend(sanpled_batch['sketch_path'])

            for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
                if positive_name not in Image_Name:
                    Image_Name.append(sanpled_batch['positive_path'][i_num])
                    Image_Feature_ALL.append(positive_feature[i_num])

        rank = torch.zeros(len(Sketch_Name))
        Image_Feature_ALL = torch.stack(Image_Feature_ALL)

        for num, sketch_feature in enumerate(Sketch_Feature_ALL):
            s_name = Sketch_Name[num]
            sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
            position_query = Image_Name.index(sketch_query_name)
            distance=cal_dis(sketch_feature.unsqueeze(0),Image_Feature_ALL)
            target_distance=cal_dis(sketch_feature.unsqueeze(0),Image_Feature_ALL[position_query].unsqueeze(0))


            #distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
            #target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
            #                                      Image_Feature_ALL[position_query].unsqueeze(0))

            rank[num] = distance.le(target_distance).sum()

        top1 = rank.le(1).sum().numpy() / rank.shape[0]
        top10 = rank.le(10).sum().numpy() / rank.shape[0]

        print('Time to EValuate:{}'.format(time.time() - start_time))
        return top1, top10

    def test_forward(self, batch):            #  this is being called only during evaluation
        sketch_feature = self.sample_embedding_network_s(batch['sketch_img'].to(device))
        positive_feature = self.sample_embedding_network_p(batch['positive_img'].to(device))
        return sketch_feature.cpu(), positive_feature.cpu()



