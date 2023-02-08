#Only use if your file names of the images and txts are identical
rm ./datasets/train_img.txt
rm ./datasets/train_gt.txt
rm ./datasets/test_img.txt
rm ./datasets/test_gt.txt
rm ./datasets/train.txt
rm ./datasets/test.txt
ls ./datasets/train/img/*.jpg > ./datasets/train_img.txt
ls ./datasets/train/gt/*.txt > ./datasets/train_gt.txt
ls ./datasets/test/img/*.jpg > ./datasets/test_img.txt
ls ./datasets/test/gt/*.txt > ./datasets/test_gt.txt
paste ./datasets/train_img.txt ./datasets/train_gt.txt > ./datasets/train.txt
paste ./datasets/test_img.txt ./datasets/test_gt.txt > ./datasets/test.txt
rm ./datasets/train_img.txt
rm ./datasets/train_gt.txt
rm ./datasets/test_img.txt
rm ./datasets/test_gt.txt
