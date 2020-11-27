package com.paddelocr_springboot.demo.Controller;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.json.JSONObject;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.ResourceUtils;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.http.*;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import sun.misc.BASE64Encoder;
import java.util.Objects;

@RestController
public class OCR {
@RequestMapping("/")
    public ResponseEntity<String> hi(){
        //创建请求头
        HttpHeaders headers = new HttpHeaders();
        //设置请求头格式
        headers.setContentType(MediaType.APPLICATION_JSON);
        //读入静态资源文件1.png
        InputStream imagePath = this.getClass().getResourceAsStream("/1.png");
        //构建请求参数
        MultiValueMap<String, String> map= new LinkedMultiValueMap<String, String>();
        //添加请求参数images，并将Base64编码的图片传入
        map.add("images", ImageToBase64(imagePath));
        //构建请求
        HttpEntity<MultiValueMap<String, String>> request = new HttpEntity<MultiValueMap<String, String>>(map, headers);
        RestTemplate restTemplate = new RestTemplate();
        //发送请求
        ResponseEntity<String> response = restTemplate.postForEntity("http://127.0.0.1:8866/predict/ocr_system", request, String.class);
        //打印请求返回值
        return response;
    }
    private String ImageToBase64(InputStream imgPath) {
        byte[] data = null;
        // 读取图片字节数组
        try {
            InputStream in = imgPath;
            System.out.println(imgPath);
            data = new byte[in.available()];
            in.read(data);
            in.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        // 对字节数组Base64编码
        BASE64Encoder encoder = new BASE64Encoder();
        // 返回Base64编码过的字节数组字符串
        //System.out.println("图片转换Base64:" + encoder.encode(Objects.requireNonNull(data)));
        return encoder.encode(Objects.requireNonNull(data));
    }

}
