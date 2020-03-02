from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os
from torch.utils import data
from os import makedirs
import torchvision
from PIL import Image
import sys
import copy
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        # self.sm1 = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # x = self.sm1(x)
        return x

def numpy_loader(input):
    item = np.load(input)/255.0
    return Image.fromarray(item)

def evaluate_model_for_accuracy(model, device, data_loader):
    model.eval()

    correct = 0
    with torch.no_grad():
        # outF = open("myOutFile.txt", "w")
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(output)
            pred = output.argmax(dim=1, keepdim=True)
            # torch.save(output, "tensor-pytorch.txt")
            # outF.write(str(output))
            # outF.write(str(pred))
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\n Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


def evaluate_adv_images(model, device, kwargs, mean, std, data_loader):
    batch_size = 100
    model.eval()

    adv_data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.DatasetFolder('adver_images', #Change this to your adv_images folder
                                           loader=numpy_loader,
                                           extensions='.npy',
                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize(mean, std)])),
                                                                         batch_size=batch_size, **kwargs)

    given_dataset = []
    adv_images = []
    labels = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            if len(given_dataset) ==0:
                given_dataset = data.squeeze().detach().cpu().numpy()
            else:
                given_dataset = np.concatenate([given_dataset, data.squeeze().detach().cpu().numpy()],
                                           axis=0)

        for data, target in adv_data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            label = target.squeeze().detach().cpu().numpy()
            softmax_values = torch.nn.Softmax()(output).cpu().numpy()[np.arange(batch_size), label]
            adv_images = data
            labels = target


    #Checking the range of generated images
    adv_images_copy = copy.deepcopy(adv_images)
    for k in range(adv_images_copy.shape[0]):
        image_ = adv_images_copy[k, :, :]

        for t, m, s in zip(image_, mean, std):
            t.mul_(s).add_(m)

        image = image_.squeeze().detach().cpu().numpy()
        image = 255.0 * image
        print(image)
        if np.min(image) < 0 or np.max(image) > 255:
            print('Generated adversarial image is out of range.')
            sys.exit()

    adv_images = adv_images.squeeze().detach().cpu().numpy()
    labels = labels.squeeze().detach().cpu().numpy()


    #Checking for equation 2 and equation 3
    if all([x > 0.8 for x in softmax_values.tolist()]):
        print('Softmax values for all of your adv images are greater than 0.8')
        S = 0
        for i in range(10):
            label_indices = np.where(labels==i)[0]
            a_i = adv_images[label_indices, :, :]
            for k in range(10):
                image = a_i[k, :, :]
                S = S + np.min(
                            np.sqrt(
                                np.sum(
                                    np.square(
                                        np.subtract(given_dataset, np.tile(np.expand_dims(image, axis=0), [1000,1,1]))
                                    ),axis=(1,2))))

        print('Value of S : {:.4f}'.format(S / 100))

    else:
        print('Softmax values for some of your adv images are less than 0.8')


import os, shutil


def remove_files_in_dir(folder):
    try:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    except Exception as e:
        print('Failed to delete. Reason: %s' % (e))

def generate_adv_images(model, device, kwargs):
    model.eval()

    # forSoftMaxList = torch.cuda.FloatTensor()
    sfMax = []

    targeted_class_labelsList = []
    image_namesList = []
    maxImages = 10
    advImageID = 1
    targetList=[]
    targetDict={}
    mean = (0.1307,)
    std = (0.3081,)

    with torch.enable_grad():

        path = 'D:/MComp/CS5260 Deep Learning and NN-2/Assignment_1/Assignment_1/data/'
        dataLoader = torch.utils.data.DataLoader(
            torchvision.datasets.DatasetFolder(path,
                                                loader=numpy_loader,
                                                extensions='.npy',
                                                transform=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])),
                                                batch_size=100, **kwargs)   #hardcoding batch size to load the class one by one

        given_dataset = []
        for data1, target1 in dataLoader:
            data1, target1 = data1.to(device), target1.to(device)
            if len(given_dataset) ==0:
                given_dataset = data1.squeeze().detach().cpu().numpy()
            else:
                given_dataset = np.concatenate([given_dataset, data1.squeeze().detach().cpu().numpy()],
                                           axis=0)

        # epsilon =0.01
        # iterNum=10
        iterNum =[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        epsilonarr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        iterNum= [i *200 for i in iterNum]
        epsilonarr = [i /10000 for i in epsilonarr]
        # epsilonarr = np.ones(10)
        # epsilonarr = epsilonarr * epsilon
        # BestvalS = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        # BestvalIter = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        # Bestepsilon = [0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008]

        for i in range(1): #Increase the range and coment out the bestval and best iter check to fin dthe optimal epsilon and iteration

            adv_images = []
            targeted_class_labels = []
            image_names = []
            print("Epsilon Value: " + str(epsilonarr))
            print("Iteration number : " + str(iterNum))
            # epsilonarr = np.ones(10)
            # epsilonarr = epsilonarr * epsilon

            targetID = 0
            totalSoftmaxList = []
            maxIndex=[]
            Sval=0
            for labelCount in range(10):
                # epsilonarr[1]=0.65
                # if labelCount == 1:
                #     iterNum=40
                adv_imagesList = torch.cuda.FloatTensor()
                softMaxList = []


                for data_f, target_f in dataLoader:
                    data, target = data_f.to(device), target_f.to(device)
                    targetCopy = torch.LongTensor(100,).zero_().to(device)
                    originalLabel = target[0].item()
                    if (originalLabel == labelCount) | (epsilonarr[labelCount] == 0):
                        # targetID+=1
                        continue
                    data.requires_grad = True

                    targetCopy += targetID

                    for name, param in model.named_parameters():
                        # print(name, param.requires_grad)
                        param.requires_grad = False
                    maxim = (1.0 - mean[0])/std[0]
                    minim = (0.0 - mean[0])/std[0]
                    data = minim + (maxim - data) #inverting images
                    def clip(data_tensor, minimum, maximum):
                        return torch.autograd.Variable(torch.clamp(data_tensor, min=minimum, max=maximum), requires_grad=True)
                        # return torch.clamp(data_tensor, min=minimum, max=maximum)

                    def norm_custom(data):
                        data_unnorm = data * std[0] + mean[0]
                        maxim = data_unnorm.max(axis=1)[0].max()
                        minim = data_unnorm.min(axis=1)[0].min()
                        return torch.autograd.Variable((data_unnorm - minim) / (maxim - minim), requires_grad=True)

                    # data_norm = clip(data, minim, maxim)
                    data_norm = torch.autograd.Variable(data.clone(), requires_grad=True)
                    iter = 0

                    while iter <= iterNum[labelCount]:
                        # print(iter+1, '->', data_norm.requires_grad)
                        model.zero_grad()
                        data_norm.grad = None
                        output = model.forward(data_norm)
                        outSoftMax = F.softmax(output, dim=1)
                        loss = F.cross_entropy(outSoftMax, targetCopy)

                        # output = model.forward(advImage)
                        loss.backward()
                        gradient = data_norm.grad.data.sign()
                        data_norm.data = data_norm.data - epsilonarr[labelCount] * gradient
                        data_norm = clip(data_norm, minim, maxim)

                        # For stopping iter if already 10 values above 0.8

                        out_max1 = outSoftMax.max(axis=1)
                        indices1 = out_max1[0].detach().cpu().numpy().argsort()
                        out_arg_max1 = out_max1[1].cpu().numpy()
                        idx1 = 99
                        image_iter1 = 0
                        counter = 0
                        while image_iter1 < 5 and idx1 >= 0:
                            if (out_arg_max1[indices1[idx1].item()].item() == targetCopy[0].item()) & (out_max1[0][indices1[idx1].item()].item() >= 0.8) :
                                image_iter1 +=1
                            idx1-=1
                        if (image_iter1 >= 5):
                            # print("Iteration stopped for " + str(originalLabel) + " at iter " + str(iter))
                            break

                        iter += 1

                    # data_norm = data.detach().clone().type(torch.cuda.IntTensor).type(torch.cuda.FloatTensor)
                    # data_norm = norm_custom(data)
                    with torch.no_grad():
                        output = model(data_norm)
                    outSoftMax = F.softmax(output, dim=1)
                    out_max = outSoftMax.max(axis=1)
                    # print(indices)
                    out_arg_max = out_max[1].cpu().numpy()
                    indices = out_max[0].detach().cpu().numpy().argsort()

                    #For finding S
                    Sarray = []
                    adv_images1 = data_norm.squeeze().detach().cpu().numpy()
                    # labels1 = out_arg_max
                    # label_indices = np.where(labels1 == labelCount)[0]
                    # a_i = adv_images1[label_indices,:,:]
                    a_i = adv_images1
                    for k in range(len(a_i)):
                        image= a_i[k,:,:]
                        Sarray.append(np.min(np.sqrt(np.sum(np.square(np.subtract(given_dataset, np.tile(np.expand_dims(image, axis=0), [1000,1,1]))),axis=(1,2)))))

                    sSortIndex = np.asarray(Sarray).argsort()

                    sValitem=0
                    image_iter = 0
                    idx = 99
                    while image_iter < 10 and idx >= 0:

                        if out_arg_max[sSortIndex[idx].item()].item() == labelCount: #Assuming the perturbed image wrongly classifies with high prob
                            adv_imagesList = torch.cat([adv_imagesList, data_norm[sSortIndex[idx].item()]], dim=0)
                            softMaxList.append(out_max[0][sSortIndex[idx].item()].item())
                            # sValitem = sValitem+ Sarray[sSortIndex[idx].item()]


                            # targeted_class_labelsList.append(originalLabel)
                            # image_namesList.append(f'sasi_{image_iter}_{out_arg_max[indices[idx].item()].item()}_{output[indices[idx].item()][out_arg_max[indices[idx].item()].item()].item()}')
                            image_iter +=1
                        idx -= 1
                    # print("Sval of " + str(labelCount) + " : " + str(sValitem/len(Sarray)))

                    # # For epsilon adjustment
                    # softmaxIndex = sorted(range(len(softMaxList)), key=softMaxList.__getitem__)
                    #
                    # if(len(softMaxList)>=10):
                    #     if softMaxList[softmaxIndex[len(softMaxList)-10]] > 0.8:
                    #         epsilonarr[labelCount] = 0
                    #         print("Epsilon stopped for " + str(labelCount))

                remove_files_in_dir(os.path.join(os.getcwd(), f'Final/Final_{labelCount}'))
                # os.mkdir(f'Final/FInal_{labelCount}')
                # for idx, img in enumerate(adv_imagesList):
                #     img_t = (img.detach().cpu().numpy() * std[0] + mean[0]) * 255
                #     im = Image.fromarray(img_t.reshape(28, 28).astype('uint8'), mode='L')
                #     im.save(f'adv_imgs_{labelCount}/i_{idx}_{softMaxList[idx]}.jpg')

                    # pixels = np.array((img * 255).cpu().detach().numpy(), dtype='int')
                    # pixels = pixels.reshape((28, 28))
                    # plt.imsave(
                    #     f'adv_imgs_{labelCount}/sasi_{idx}_{softMaxList[idx]}.jpeg',
                    #     pixels)



                #Sort the image list adn indices
                adv_imagesList4dim = adv_imagesList[:,None,:, :]
                with torch.no_grad():
                    outputAdvList = model(adv_imagesList4dim)
                outSoftMaxAdvList = F.softmax(outputAdvList, dim=1)
                outAdvMax =  outSoftMaxAdvList.max(axis=1)
                indicesAdvList = outAdvMax[0].detach().cpu().numpy().argsort()
                outAdvArgMax = outAdvMax[1].cpu().numpy()

                for idx, img in enumerate(adv_imagesList):
                    img_t = (img.detach().cpu().numpy() * std[0] + mean[0]) * 255
                    im = Image.fromarray(img_t.reshape(28, 28).astype('uint8'), mode='L')
                    im.save(f'Final/Final_{labelCount}/i_{idx}_{outAdvMax[0][idx].item()}_{outAdvArgMax[idx].item()}.jpg')

                #For finding final 10 based on S score
                FinalSarray = []
                Finaladv_images1 = adv_imagesList4dim.squeeze().detach().cpu().numpy()
                # labels1 = out_arg_max
                # label_indices = np.where(labels1 == labelCount)[0]
                # a_i = adv_images1[label_indices,:,:]
                Finala_i = Finaladv_images1
                for k in range(len(Finala_i)):
                    Finalimage = Finala_i[k, :, :]
                    FinalSarray.append(np.min(np.sqrt(
                        np.sum(np.square(np.subtract(given_dataset, np.tile(np.expand_dims(Finalimage, axis=0), [1000, 1, 1]))),
                               axis=(1, 2)))))

                FinalsSortIndex = np.asarray(FinalSarray).argsort()

                image_iter_advlist = 0
                idx_advList = len(adv_imagesList4dim)-1
                SvalIndividual= 0
                while image_iter_advlist < 10 and idx_advList >= 0:

                    if (outAdvArgMax[FinalsSortIndex[idx_advList].item()].item() == labelCount) & (outAdvMax[0][FinalsSortIndex[idx_advList].item()].item() > 0.8):
                        adv_images.append(adv_imagesList4dim[FinalsSortIndex[idx_advList]])
                        totalSoftmaxList.append(outAdvMax[0][FinalsSortIndex[idx_advList].item()].item())
                        maxIndex.append(outAdvMax[1][FinalsSortIndex[idx_advList].item()].item())
                        Sval = Sval + FinalSarray[FinalsSortIndex[idx_advList].item()]
                        SvalIndividual =SvalIndividual + FinalSarray[FinalsSortIndex[idx_advList].item()]
                        # with torch.no_grad():
                        #     sfMax.append((F.softmax(model(adv_imagesList4dim[indicesAdvList[idx_advList]][:,None,:,:]), dim=1)).max(axis=1)[0].item())
                        # forSoftMaxList = torch.cat([forSoftMaxList, adv_imagesList4dim[indicesAdvList[idx_advList].item()]], dim=0)
                        targeted_class_labels.append(labelCount)
                        image_names.append(
                            f'Final_{image_iter_advlist}_{outAdvArgMax[FinalsSortIndex[idx_advList].item()].item()}_{outSoftMaxAdvList[FinalsSortIndex[idx_advList].item()][outAdvArgMax[FinalsSortIndex[idx].item()].item()].item()*100:{1}.{4}}')
                        image_iter_advlist += 1
                    idx_advList -= 1
                # print("Minimum Softmax of " + str(labelCount) + ":" + str(
                #     min(totalSoftmaxList[10 * labelCount:10 * labelCount + 10])))
                # print("Maximum Softmax of " + str(labelCount) + ":" + str(
                #     max(totalSoftmaxList[10 * labelCount:10 * labelCount + 10])))
                print(SvalIndividual/10)
                # if BestvalS[labelCount] < SvalIndividual:
                #     BestvalS[labelCount] = SvalIndividual
                #     # BestvalIter[labelCount] = iterNum[labelCount]
                #     Bestepsilon[labelCount] = epsilonarr[labelCount]
                targetID+=1    # break

                # print("Done -> "+str(labelCount))
            # forSoftMaxList = forSoftMaxList[:,None, :, :]
            # with torch.no_grad():
            #     SMoutputAdvList = model(forSoftMaxList)
            # softMaxValsFinal = F.softmax(SMoutputAdvList, dim=1)
            # sFmAX = softMaxValsFinal.max(axis=1)[0]
            print("S Value: " + str(Sval/len(maxIndex)))
            if len(adv_images) < 100:
                print("Not enough images generated")
            # epsilonarr = [i + 0.001 for i in epsilonarr]
    # print("Best S val: " + str(BestvalS))
    # print("Best iter: " + str(BestvalIter))
    return adv_images,image_names,targeted_class_labels

def main():
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--model_path', type=str, default='model/mnist_cnn.pt')
    parser.add_argument('--data_folder', type=str, default='data')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    mean = (0.1307,)
    std = (0.3081,)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    data_loader =  torch.utils.data.DataLoader(
        torchvision.datasets.DatasetFolder('D:/MComp/CS5260 Deep Learning and NN-2/Assignment_1/Assignment_1/data',
                                           loader= numpy_loader,
                                           extensions= '.npy',
                                           transform=transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean, std)])),
                                           batch_size=args.batch_size, **kwargs)

    model = Net().to(device)

    model.load_state_dict(torch.load(args.model_path))

    evaluate_model_for_accuracy(model, device, data_loader)

    adv_images,image_names,class_labels = generate_adv_images(model, device, kwargs)
    #Implement this method to generate adv images
    #statisfying constraints mentioned in the assignment discription

    save_folder = 'adver_images'
    remove_files_in_dir(os.path.join(save_folder))

    for image,image_name,class_label in zip(adv_images,image_names,class_labels):
        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)

        image_to_save = image.squeeze().detach().cpu().numpy()
        image_to_save = 255.0 * image_to_save

        if np.min(image_to_save) < 0 or np.max(image_to_save) > 255:
            print('Generated adversarial image is out of range.')
            sys.exit()
        if not os.path.exists(os.path.join(save_folder,str(class_label))):
            makedirs(os.path.join(save_folder,str(class_label)))

        np.save(os.path.join(save_folder,str(class_label),image_name), image_to_save)

    evaluate_adv_images(model,device,kwargs,mean,std,data_loader)


if __name__ == '__main__':
    main()
