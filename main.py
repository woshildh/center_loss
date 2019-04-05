import torch
import model,center_loss,utils,dataloader
import torch.nn.functional as F
import tensorboardX
import numpy as np
import torch.optim.lr_scheduler as lr_sche

#设定一些超参数
weight_center=1
batch_size=64
epochs=100
classes_num=10
model_lr=0.001
center_lr=0.1
use_gpu=True
display_step=40

def train(net,net_center,optimizer_model,optimizer_centerloss,trainloader,
    epoch,use_gpu):
    
    loss1 = utils.AverageMeter()
    loss2 = utils.AverageMeter()
    loss3 = utils.AverageMeter()
    acc = utils.AverageMeter()

    #用于收集所有的特征和标签
    all_feature,all_labels=[],[]
    for i,(x,y) in enumerate(trainloader):
        if use_gpu:
            x , y=x.cuda() , y.cuda()
        y_,feature = net(x)
        #计算损失
        loss_cross = F.cross_entropy(y_,y)
        loss_center = net_center(feature,y)
        total_loss = loss_cross + loss_center * weight_center
        #梯度清0
        optimizer_model.zero_grad()
        optimizer_centerloss.zero_grad()
        #损失更新
        total_loss.backward()
        #模型更新
        optimizer_model.step()
        optimizer_centerloss.step()
        #计算正确率
        predicts = torch.argmax(y_,dim=1,keepdim=False)
        correct_num = (predicts==y).sum().item()
        step_acc = correct_num/x.size(0)
        #记录这些值
        loss1.update(loss_cross.item())
        loss2.update(loss_center.item())
        loss3.update(total_loss.item())
        acc.update(step_acc)
        #收集结果
        all_feature.append(feature.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
        #输出结果
        if i % display_step==0:
            print("{} epoch, {} step,step acc is {:.4f}, corss loss is {:.5f}, center loss is {:.5f}".format(
                epoch,i,step_acc,loss_cross,loss_center
            ))
    #将收集的特征和label合并
    all_feature = np.concatenate(all_feature,axis=0)
    all_labels = np.concatenate(all_labels,axis=0)
    #返回结果
    return acc.avg,loss1.avg,loss2.avg,loss3.avg,all_feature,all_labels

def test(net,testloader,epoch,use_gpu):
    acc = utils.AverageMeter()
    all_feature,all_labels=[],[]
    #用于收集所有的特征和标签
    all_feature,all_labels=[],[]
    for i,(x,y) in enumerate(testloader):
        if use_gpu:
            x , y=x.cuda() , y.cuda()
        y_,feature = net(x)
        #计算正确率
        predicts = torch.argmax(y_,dim=1,keepdim=False)
        correct_num = (predicts==y).sum().item()
        step_acc = correct_num/x.size(0)
        #记录这些值
        acc.update(step_acc)
        #收集结果
        all_feature.append(feature.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
    #将收集的特征和label合并
    all_feature = np.concatenate(all_feature,axis=0)
    all_labels = np.concatenate(all_labels,axis=0)
    #返回结果
    return acc.avg,all_feature,all_labels

def main():
    net = model.ConvNet(classes_num)
    net_center = center_loss.CenterLoss(feature_dim=2,classes_num=classes_num,
        use_gpu=use_gpu)
    if use_gpu:
        net=net.cuda()
        net_center=net_center.cuda()
    optimizer_model = torch.optim.SGD(net.parameters(),lr=model_lr,weight_decay=1e-5,
        nesterov=True,momentum=0.9)
    optimizer_centerloss = torch.optim.SGD(net_center.parameters(),lr=center_lr)
    print("net and optimzer load succeed")
    trainloader , testloader = dataloader.get_loader(batch_size=batch_size,
         root_path="./data")
    print("data load succeed")
    logger = utils.Logger(tb_path="./logs/tblog/")
    scheduler = lr_sche.StepLR(optimizer_model,30,0.1)
    best_acc=0
    #开始训练
    for i in range(1,epochs+1):
        scheduler.step(epoch=i)
        model.train()
        train_acc,train_cross_loss,train_center_loss,\
            train_loss,all_feature,all_labels=\
            train(net,net_center,optimizer_model,optimizer_centerloss,\
            trainloader,i,use_gpu)
        utils.plot_features(all_feature,all_labels,classes_num,i,"./logs/images/train/train_{}.png")
        model.eval()
        test_acc,all_feature,all_labels=test(net,testloader,i,use_gpu)
        utils.plot_features(all_feature,all_labels,classes_num,i,"./logs/images/test/test_{}.png")
        print("{} epoch end, train acc is {:.4f}, test acc is {:.4f}".format(
            i,train_acc,test_acc))
        content={"Train/acc":train_acc,"Test/acc":test_acc,
            "Train/cross_loss":train_cross_loss,
            "Train/total_loss":train_loss,
            "Train/center_loss":train_center_loss,
        }
        logger.log(step=i,content=content)
        if best_acc<test_acc:
            best_acc=test_acc
        utils.save_checkpoints("./logs/weights/net_{}.pth",i,\
            net.state_dict(),(best_acc==test_acc))
        utils.save_checkpoints("./logs/weights/center_{}.pth",i,\
            net_center.state_dict() ,(best_acc==test_acc))
    utils.make_gif("./logs/images/train/","./logs/train.gif")
    utils.make_gif("./logs/images/test/","./logs/test.gif")
    print("Traing finished...")

if __name__=="__main__":
    main()
