import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    def __init__(self,feature_dim=2,classes_num=10,use_gpu=True):
        super(CenterLoss,self).__init__()
        self.feature_dim = feature_dim
        self.classes_num = classes_num
        self.use_gpu = use_gpu

        if use_gpu:
            self.centers = nn.Parameter(torch.randn(1,classes_num,feature_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(1,classes_num,feature_dim))            

    def forward(self,x,labels):
        batch_size = x.size(0)
        #改变x的形状
        x = x.unsqueeze(dim=1).expand(batch_size,self.classes_num,self.feature_dim)
        #计算dismat
        dismat = torch.pow(x-self.centers,2).sum(dim=2)
        #计算mask
        classes = torch.arange(self.classes_num).unsqueeze(0)
        labels = labels.unsqueeze(1).expand(batch_size, self.classes_num)
        if self.use_gpu:
            classes=classes.cuda()
        mask = torch.eq(labels,classes.expand(batch_size,self.classes_num)).float()
        #计算损失
        loss = (dismat*mask).sum()/batch_size
        return loss
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.classes_num) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.classes_num, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.classes_num).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.classes_num)
        mask = labels.eq(classes.expand(batch_size, self.classes_num)).float()

        dist = distmat * mask
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

if __name__=="__main__":
    feature = torch.randn(4,2).cuda()
    labels = torch.Tensor([1,2,4,7]).long().cuda()
    c=CenterLoss(2,10,True)
    l=c(feature,labels)
    print(l.item())
    l.backward()