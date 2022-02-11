# coding:utf8
import os
import shutil
import random
import argparse


# 删除划分的训练集、验证集、测试集文件夹，重新创建一个空的文件夹
def isCreateOrDeleteFolder(path, flag):
    flagPath = os.path.join(path, flag)

    if os.path.exists(flagPath):
        shutil.rmtree(flagPath)

    os.makedirs(flagPath)
    flagAbsPath = os.path.abspath(flagPath)
    return flagAbsPath


def splitTrainVal(root, absTrainRootPath, absValRootPath, absTestRootPath, trainTxt, valTxt, testTxt, flag):
    # 按照指定的比例划分训练集、验证集、测试集
    dataAbsPath = os.path.abspath(root)

    if flag == "det":
        labelFilePath = os.path.join(dataAbsPath, args.detLabelFileName)
    elif flag == "rec":
        labelFilePath = os.path.join(dataAbsPath, args.recLabelFileName)

    labelFileRead = open(labelFilePath, "r", encoding="UTF-8")
    labelFileContent = labelFileRead.readlines()
    random.shuffle(labelFileContent)
    labelRecordLen = len(labelFileContent)

    for index, labelRecordInfo in enumerate(labelFileContent):
        imageRelativePath = labelRecordInfo.split('\t')[0]
        imageLabel = labelRecordInfo.split('\t')[1]
        imageName = os.path.basename(imageRelativePath)

        if flag == "det":
            imagePath = os.path.join(dataAbsPath, imageName)
        elif flag == "rec":
            imagePath = os.path.join(dataAbsPath, "{}\\{}".format(args.recImageDirName, imageName))

        # 按预设的比例划分训练集、验证集、测试集
        trainValTestRatio = args.trainValTestRatio.split(":")
        trainRatio = eval(trainValTestRatio[0]) / 10
        valRatio = trainRatio + eval(trainValTestRatio[1]) / 10
        curRatio = index / labelRecordLen

        if curRatio < trainRatio:
            imageCopyPath = os.path.join(absTrainRootPath, imageName)
            shutil.copy(imagePath, imageCopyPath)
            trainTxt.write("{}\t{}".format(imageCopyPath, imageLabel))
        elif curRatio >= trainRatio and curRatio < valRatio:
            imageCopyPath = os.path.join(absValRootPath, imageName)
            shutil.copy(imagePath, imageCopyPath)
            valTxt.write("{}\t{}".format(imageCopyPath, imageLabel))
        else:
            imageCopyPath = os.path.join(absTestRootPath, imageName)
            shutil.copy(imagePath, imageCopyPath)
            testTxt.write("{}\t{}".format(imageCopyPath, imageLabel))


# 删掉存在的文件
def removeFile(path):
    if os.path.exists(path):
        os.remove(path)


def genDetRecTrainVal(args):
    detAbsTrainRootPath = isCreateOrDeleteFolder(args.detRootPath, "train")
    detAbsValRootPath = isCreateOrDeleteFolder(args.detRootPath, "val")
    detAbsTestRootPath = isCreateOrDeleteFolder(args.detRootPath, "test")
    recAbsTrainRootPath = isCreateOrDeleteFolder(args.recRootPath, "train")
    recAbsValRootPath = isCreateOrDeleteFolder(args.recRootPath, "val")
    recAbsTestRootPath = isCreateOrDeleteFolder(args.recRootPath, "test")

    removeFile(os.path.join(args.detRootPath, "train.txt"))
    removeFile(os.path.join(args.detRootPath, "val.txt"))
    removeFile(os.path.join(args.detRootPath, "test.txt"))
    removeFile(os.path.join(args.recRootPath, "train.txt"))
    removeFile(os.path.join(args.recRootPath, "val.txt"))
    removeFile(os.path.join(args.recRootPath, "test.txt"))

    detTrainTxt = open(os.path.join(args.detRootPath, "train.txt"), "a", encoding="UTF-8")
    detValTxt = open(os.path.join(args.detRootPath, "val.txt"), "a", encoding="UTF-8")
    detTestTxt = open(os.path.join(args.detRootPath, "test.txt"), "a", encoding="UTF-8")
    recTrainTxt = open(os.path.join(args.recRootPath, "train.txt"), "a", encoding="UTF-8")
    recValTxt = open(os.path.join(args.recRootPath, "val.txt"), "a", encoding="UTF-8")
    recTestTxt = open(os.path.join(args.recRootPath, "test.txt"), "a", encoding="UTF-8")

    splitTrainVal(args.datasetRootPath, detAbsTrainRootPath, detAbsValRootPath, detAbsTestRootPath, detTrainTxt, detValTxt,
                  detTestTxt, "det")

    for root, dirs, files in os.walk(args.datasetRootPath):
        for dir in dirs:
            if dir == 'crop_img':
                splitTrainVal(root, recAbsTrainRootPath, recAbsValRootPath, recAbsTestRootPath, recTrainTxt, recValTxt,
                              recTestTxt, "rec")
            else:
                continue
        break



if __name__ == "__main__":
    # 功能描述：分别划分检测和识别的训练集、验证集、测试集
    # 说明：可以根据自己的路径和需求调整参数，图像数据往往多人合作分批标注，每一批图像数据放在一个文件夹内用PPOCRLabel进行标注，
    # 如此会有多个标注好的图像文件夹汇总并划分训练集、验证集、测试集的需求
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainValTestRatio",
        type=str,
        default="6:2:2",
        help="ratio of trainset:valset:testset")
    parser.add_argument(
        "--datasetRootPath",
        type=str,
        default="../train_data/",
        help="path to the dataset marked by ppocrlabel, E.g, dataset folder named 1,2,3..."
    )
    parser.add_argument(
        "--detRootPath",
        type=str,
        default="../train_data/det",
        help="the path where the divided detection dataset is placed")
    parser.add_argument(
        "--recRootPath",
        type=str,
        default="../train_data/rec",
        help="the path where the divided recognition dataset is placed"
    )
    parser.add_argument(
        "--detLabelFileName",
        type=str,
        default="Label.txt",
        help="the name of the detection annotation file")
    parser.add_argument(
        "--recLabelFileName",
        type=str,
        default="rec_gt.txt",
        help="the name of the recognition annotation file"
    )
    parser.add_argument(
        "--recImageDirName",
        type=str,
        default="crop_img",
        help="the name of the folder where the cropped recognition dataset is located"
    )
    args = parser.parse_args()
    genDetRecTrainVal(args)
