<?php
$url = 'http://127.0.0.1:8866/predict/ocr_system';
$file = "./1.png";

if($fp = fopen($file,"rb", 0))
{
    $gambar = fread($fp,filesize($file));
    fclose($fp);
    $base64 = base64_encode($gambar);
}   
//由于php  json_encode()函数转换问题无法转换[]格式所以采用手动拼接
$data = '{"images":["'.$base64.'"]}';
$ch = curl_init();
curl_setopt($ch, CURLOPT_HTTPHEADER, array(
                'Content-Type: application/json; charset=utf-8',
        ));
curl_setopt($ch, CURLOPT_URL, $url);//要访问的地址
curl_setopt($ch, CURLOPT_RETURNTRANSFER, 0);//执行结果是否被返回，0是返回，1是不返回
curl_setopt($ch, CURLOPT_POST, 1);// 发送一个常规的POST请求
curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
$output = curl_exec($ch);//执行并获取数据
curl_close($ch);