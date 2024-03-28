#!/bin/bash

if [ "$#" -eq 1 ]; then
    outputPath="$1/doc3d"
else
    outputPath="$HOME/Downloads"
fi

if ! [ -x "$(command -v wget)" ]; then
    echo "Error!: wget is not installed! Please install it and try again"
    exit 1
fi

echo -e "\n### ------------------------------------------------------- ###\n"
echo "### Downloading into $outputPath"
echo -e "\n### ------------------------------------------------------- ###\n"

doc3d_download() {
    local url=$1
    local path=$2
    local files=$3
    local uname=""    # put your username !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    local pass=""     # put your password !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    echo -ne "### Downloading "$files" ###\t\n"
    wget --continue --user "$uname" --password "$pass" --directory-prefix="$path" "$url" 2>&1
    echo -ne "\b\b\b\b"
    echo " # done"
}

doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_1.zip" "$outputPath/" "img_1.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_2.zip" "$outputPath/" "img_2.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_3.zip" "$outputPath/" "img_3.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_4.zip" "$outputPath/" "img_4.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_5.zip" "$outputPath/" "img_5.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_6.zip" "$outputPath/" "img_6.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_7.zip" "$outputPath/" "img_7.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_8.zip" "$outputPath/" "img_8.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_9.zip" "$outputPath/" "img_9.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_10.zip" "$outputPath/" "img_10.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_11.zip" "$outputPath/" "img_11.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_12.zip" "$outputPath/" "img_12.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_13.zip" "$outputPath/" "img_13.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_14.zip" "$outputPath/" "img_14.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_15.zip" "$outputPath/" "img_15.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_16.zip" "$outputPath/" "img_16.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_17.zip" "$outputPath/" "img_17.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_18.zip" "$outputPath/" "img_18.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_19.zip" "$outputPath/" "img_19.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_20.zip" "$outputPath/" "img_20.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/img_21.zip" "$outputPath/" "img_21.zip"

doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_1.zip" "$outputPath/" "wc_1.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_2.zip" "$outputPath/" "wc_2.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_3.zip" "$outputPath/" "wc_3.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_4.zip" "$outputPath/" "wc_4.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_5.zip" "$outputPath/" "wc_5.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_6.zip" "$outputPath/" "wc_6.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_7.zip" "$outputPath/" "wc_7.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_8.zip" "$outputPath/" "wc_8.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_9.zip" "$outputPath/" "wc_9.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_10.zip" "$outputPath/" "wc_10.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_11.zip" "$outputPath/" "wc_11.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_12.zip" "$outputPath/" "wc_12.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_13.zip" "$outputPath/" "wc_13.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_14.zip" "$outputPath/" "wc_14.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_15.zip" "$outputPath/" "wc_15.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_16.zip" "$outputPath/" "wc_16.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_17.zip" "$outputPath/" "wc_17.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_18.zip" "$outputPath/" "wc_18.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_19.zip" "$outputPath/" "wc_19.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_20.zip" "$outputPath/" "wc_20.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/wc_21.zip" "$outputPath/" "wc_21.zip"

doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_1.zip" "$outputPath/" "bm_1.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_2.zip" "$outputPath/" "bm_2.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_3.zip" "$outputPath/" "bm_3.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_4.zip" "$outputPath/" "bm_4.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_5.zip" "$outputPath/" "bm_5.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_6.zip" "$outputPath/" "bm_6.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_7.zip" "$outputPath/" "bm_7.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_8.zip" "$outputPath/" "bm_8.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_9.zip" "$outputPath/" "bm_9.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_10.zip" "$outputPath/" "bm_10.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_11.zip" "$outputPath/" "bm_11.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_12.zip" "$outputPath/" "bm_12.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_13.zip" "$outputPath/" "bm_13.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_14.zip" "$outputPath/" "bm_14.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_15.zip" "$outputPath/" "bm_15.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_16.zip" "$outputPath/" "bm_16.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_17.zip" "$outputPath/" "bm_17.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_18.zip" "$outputPath/" "bm_18.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_19.zip" "$outputPath/" "bm_19.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_20.zip" "$outputPath/" "bm_20.zip"
doc3d_download "http://vision.cs.stonybrook.edu/~sagnik/doc3d/bm_21.zip" "$outputPath/" "bm_21.zip"


echo -e "\n### ------------------------------------------------------- ###\n"
echo "### Unzipping downloaded files ###"
echo -e "\n### ------------------------------------------------------- ###\n"
echo -e $outputPath"/img_1.zip .."
unzip -q $outputPath"/img_1.zip" -d $outputPath
rm $outputPath"/img_1.zip"
echo -e $outputPath"/img_2.zip .."
unzip -q $outputPath"/img_2.zip" -d $outputPath
rm $outputPath"/img_2.zip"
echo -e $outputPath"/img_3.zip .."
unzip -q $outputPath"/img_3.zip" -d $outputPath
rm $outputPath"/img_3.zip"
echo -e $outputPath"/img_4.zip .."
unzip -q $outputPath"/img_4.zip" -d $outputPath
rm $outputPath"/img_4.zip"
echo -e $outputPath"/img_5.zip .."
unzip -q $outputPath"/img_5.zip" -d $outputPath
rm $outputPath"/img_5.zip"
echo -e $outputPath"/img_6.zip .."
unzip -q $outputPath"/img_6.zip" -d $outputPath
rm $outputPath"/img_6.zip"
echo -e $outputPath"/img_7.zip .."
unzip -q $outputPath"/img_7.zip" -d $outputPath
rm $outputPath"/img_7.zip"
echo -e $outputPath"/img_8.zip .."
unzip -q $outputPath"/img_8.zip" -d $outputPath
rm $outputPath"/img_8.zip"
echo -e $outputPath"/img_9.zip .."
unzip -q $outputPath"/img_9.zip" -d $outputPath
rm $outputPath"/img_9.zip"
echo -e $outputPath"/img_10.zip .."
unzip -q $outputPath"/img_10.zip" -d $outputPath
rm $outputPath"/img_10.zip"
echo -e $outputPath"/img_11.zip .."
unzip -q $outputPath"/img_11.zip" -d $outputPath
rm $outputPath"/img_11.zip"
echo -e $outputPath"/img_12.zip .."
unzip -q $outputPath"/img_12.zip" -d $outputPath
rm $outputPath"/img_12.zip"
echo -e $outputPath"/img_13.zip .."
unzip -q $outputPath"/img_13.zip" -d $outputPath
rm $outputPath"/img_13.zip"
echo -e $outputPath"/img_14.zip .."
unzip -q $outputPath"/img_14.zip" -d $outputPath
rm $outputPath"/img_14.zip"
echo -e $outputPath"/img_15.zip .."
unzip -q $outputPath"/img_15.zip" -d $outputPath
rm $outputPath"/img_15.zip"
echo -e $outputPath"/img_16.zip .."
unzip -q $outputPath"/img_16.zip" -d $outputPath
rm $outputPath"/img_16.zip"
echo -e $outputPath"/img_17.zip .."
unzip -q $outputPath"/img_17.zip" -d $outputPath
rm $outputPath"/img_17.zip"
echo -e $outputPath"/img_18.zip .."
unzip -q $outputPath"/img_18.zip" -d $outputPath
rm $outputPath"/img_18.zip"
echo -e $outputPath"/img_19.zip .."
unzip -q $outputPath"/img_19.zip" -d $outputPath
rm $outputPath"/img_19.zip"
echo -e $outputPath"/img_20.zip .."
unzip -q $outputPath"/img_20.zip" -d $outputPath
rm $outputPath"/img_20.zip"
echo -e $outputPath"/img_21.zip .."
unzip -q $outputPath"/img_21.zip" -d $outputPath
rm $outputPath"/img_21.zip"

echo -e $outputPath"/wc_1.zip .."
unzip -q $outputPath"/wc_1.zip" -d $outputPath
rm $outputPath"/wc_1.zip"
echo -e $outputPath"/wc_2.zip .."
unzip -q $outputPath"/wc_2.zip" -d $outputPath
rm $outputPath"/wc_2.zip"
echo -e $outputPath"/wc_3.zip .."
unzip -q $outputPath"/wc_3.zip" -d $outputPath
rm $outputPath"/wc_3.zip"
echo -e $outputPath"/wc_4.zip .."
unzip -q $outputPath"/wc_4.zip" -d $outputPath
rm $outputPath"/wc_4.zip"
echo -e $outputPath"/wc_5.zip .."
unzip -q $outputPath"/wc_5.zip" -d $outputPath
rm $outputPath"/wc_5.zip"
echo -e $outputPath"/wc_6.zip .."
unzip -q $outputPath"/wc_6.zip" -d $outputPath
rm $outputPath"/wc_6.zip"
echo -e $outputPath"/wc_7.zip .."
unzip -q $outputPath"/wc_7.zip" -d $outputPath
rm $outputPath"/wc_7.zip"
echo -e $outputPath"/wc_8.zip .."
unzip -q $outputPath"/wc_8.zip" -d $outputPath
rm $outputPath"/wc_8.zip"
echo -e $outputPath"/wc_9.zip .."
unzip -q $outputPath"/wc_9.zip" -d $outputPath
rm $outputPath"/wc_9.zip"
echo -e $outputPath"/wc_10.zip .."
unzip -q $outputPath"/wc_10.zip" -d $outputPath
rm $outputPath"/wc_10.zip"
echo -e $outputPath"/wc_11.zip .."
unzip -q $outputPath"/wc_11.zip" -d $outputPath
rm $outputPath"/wc_11.zip"
echo -e $outputPath"/wc_12.zip .."
unzip -q $outputPath"/wc_12.zip" -d $outputPath
rm $outputPath"/wc_12.zip"
echo -e $outputPath"/wc_13.zip .."
unzip -q $outputPath"/wc_13.zip" -d $outputPath
rm $outputPath"/wc_13.zip"
echo -e $outputPath"/wc_14.zip .."
unzip -q $outputPath"/wc_14.zip" -d $outputPath
rm $outputPath"/wc_14.zip"
echo -e $outputPath"/wc_15.zip .."
unzip -q $outputPath"/wc_15.zip" -d $outputPath
rm $outputPath"/wc_15.zip"
echo -e $outputPath"/wc_16.zip .."
unzip -q $outputPath"/wc_16.zip" -d $outputPath
rm $outputPath"/wc_16.zip"
echo -e $outputPath"/wc_17.zip .."
unzip -q $outputPath"/wc_17.zip" -d $outputPath
rm $outputPath"/wc_17.zip"
echo -e $outputPath"/wc_18.zip .."
unzip -q $outputPath"/wc_18.zip" -d $outputPath
rm $outputPath"/wc_18.zip"
echo -e $outputPath"/wc_19.zip .."
unzip -q $outputPath"/wc_19.zip" -d $outputPath
rm $outputPath"/wc_19.zip"
echo -e $outputPath"/wc_20.zip .."
unzip -q $outputPath"/wc_20.zip" -d $outputPath
rm $outputPath"/wc_20.zip"
echo -e $outputPath"/wc_21.zip .."
unzip -q $outputPath"/wc_21.zip" -d $outputPath
rm $outputPath"/wc_21.zip"

echo -e $outputPath"/bm_1.zip .."
unzip -q $outputPath"/bm_1.zip" -d $outputPath
rm $outputPath"/bm_1.zip"
echo -e $outputPath"/bm_2.zip .."
unzip -q $outputPath"/bm_2.zip" -d $outputPath
rm $outputPath"/bm_2.zip"
echo -e $outputPath"/bm_3.zip .."
unzip -q $outputPath"/bm_3.zip" -d $outputPath
rm $outputPath"/bm_3.zip"
echo -e $outputPath"/bm_4.zip .."
unzip -q $outputPath"/bm_4.zip" -d $outputPath
rm $outputPath"/bm_4.zip"
echo -e $outputPath"/bm_5.zip .."
unzip -q $outputPath"/bm_5.zip" -d $outputPath
rm $outputPath"/bm_5.zip"
echo -e $outputPath"/bm_6.zip .."
unzip -q $outputPath"/bm_6.zip" -d $outputPath
rm $outputPath"/bm_6.zip"
echo -e $outputPath"/bm_7.zip .."
unzip -q $outputPath"/bm_7.zip" -d $outputPath
rm $outputPath"/bm_7.zip"
echo -e $outputPath"/bm_8.zip .."
unzip -q $outputPath"/bm_8.zip" -d $outputPath
rm $outputPath"/bm_8.zip"
echo -e $outputPath"/bm_9.zip .."
unzip -q $outputPath"/bm_9.zip" -d $outputPath
rm $outputPath"/bm_9.zip"
echo -e $outputPath"/bm_10.zip .."
unzip -q $outputPath"/bm_10.zip" -d $outputPath
rm $outputPath"/bm_10.zip"
echo -e $outputPath"/bm_11.zip .."
unzip -q $outputPath"/bm_11.zip" -d $outputPath
rm $outputPath"/bm_11.zip"
echo -e $outputPath"/bm_12.zip .."
unzip -q $outputPath"/bm_12.zip" -d $outputPath
rm $outputPath"/bm_12.zip"
echo -e $outputPath"/bm_13.zip .."
unzip -q $outputPath"/bm_13.zip" -d $outputPath
rm $outputPath"/bm_13.zip"
echo -e $outputPath"/bm_14.zip .."
unzip -q $outputPath"/bm_14.zip" -d $outputPath
rm $outputPath"/bm_14.zip"
echo -e $outputPath"/bm_15.zip .."
unzip -q $outputPath"/bm_15.zip" -d $outputPath
rm $outputPath"/bm_15.zip"
echo -e $outputPath"/bm_16.zip .."
unzip -q $outputPath"/bm_16.zip" -d $outputPath
rm $outputPath"/bm_16.zip"
echo -e $outputPath"/bm_17.zip .."
unzip -q $outputPath"/bm_17.zip" -d $outputPath
rm $outputPath"/bm_17.zip"
echo -e $outputPath"/bm_18.zip .."
unzip -q $outputPath"/bm_18.zip" -d $outputPath
rm $outputPath"/bm_18.zip"
echo -e $outputPath"/bm_19.zip .."
unzip -q $outputPath"/bm_19.zip" -d $outputPath
rm $outputPath"/bm_19.zip"
echo -e $outputPath"/bm_20.zip .."
unzip -q $outputPath"/bm_20.zip" -d $outputPath
rm $outputPath"/bm_20.zip"
echo -e $outputPath"/bm_21.zip .."
unzip -q $outputPath"/bm_21.zip" -d $outputPath
rm $outputPath"/bm_21.zip"

echo -e "\n### ------------------------------------------------------- ###\n"
echo "### All done!"
echo -e "\n### ------------------------------------------------------- ###\n"
