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
//Because the PHP json_encode() function conversion problem was unable to convert the [] format, manual splicers were used
$data = '{"images":["'.$base64.'"]}';
$ch = curl_init();
curl_setopt($ch, CURLOPT_HTTPHEADER, array(
                'Content-Type: application/json; charset=utf-8',
        ));
//要访问的地址
//The address to visit
curl_setopt($ch, CURLOPT_URL, $url);
//执行结果是否被返回，0是返回，1是不返回
//Whether the execution result is returned, 0 is returned, 1 is not returned
curl_setopt($ch, CURLOPT_RETURNTRANSFER, 0);
// 发送一个常规的POST请求
//Send a regular POST request
curl_setopt($ch, CURLOPT_POST, 1);
curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
//执行并获取数据
//Execute and retrieve data
$output = curl_exec($ch);
curl_close($ch);