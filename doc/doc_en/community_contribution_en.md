# Community Contribution

Thank you for your support and interest in PaddleOCR. The goal of  PaddleOCR is to build a professional, harmonious and supportive open source community with developers. This document presents existing community contributions, explanations for various contributions, and new opportunities and processes to make the contribution process more efficient and clear.

PaddleOCR wants to help any developer with a dream realize their vision and enjoy the joy of creating value through the power of AI.

---

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR" />
</a>

> The picture above shows PaddleOCR's current Contributor, updated regularly

## 1. Community Contribution 

### 1.1 PaddleOCR Based Community Project

- 【The lastest】 [FastOCRLabel](https://gitee.com/BaoJianQiang/FastOCRLabel): Complete C# version annotation tool (@ [包建强](https://gitee.com/BaoJianQiang) )

#### 1.1.1 Universal Tools

- [DangoOCR offline version](https://github.com/PantsuDango/DangoOCR)：Universal desktop instant translation tool (@ [PantsuDango](https://github.com/PantsuDango))
- [scr2txt](https://github.com/lstwzd/scr2txt)：Screenshot to Text tool (@ [lstwzd](https://github.com/lstwzd))
- [AI Studio project](https://aistudio.baidu.com/aistudio/projectdetail/1054614?channelType=0&channel=0)：English video automatically generates subtitles( @ [叶月水狐](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/322052))

#### 1.1.2 Vertical Scene Tools

- [id_card_ocr](https://github.com/baseli/id_card_ocr)：Identification of copy of ID card(@ [baseli](https://github.com/baseli))
- [Paddle_Table_Image_Reader](https://github.com/thunder95/Paddle_Table_Image_Reader): A data assistant that can read tables and pictures(@ [thunder95](https://github.com/thunder95]))

#### 1.1.3 Pre And Post Processing

- [paddleOCRCorrectOutputs](https://github.com/yuranusduke/paddleOCRCorrectOutputs)：Get the key-value of OCR recognition result (@ [yuranusduke](https://github.com/yuranusduke))

### 1.2 New Features For PaddleOCR

- Thanks [authorfu](https://github.com/authorfu) for contributing Android([#340](https://github.com/PaddlePaddle/PaddleOCR/pull/340)) and [xiadeye](https://github.com/xiadeye) for contributing IOS demo code([#325](https://github.com/PaddlePaddle/PaddleOCR/pull/325)).
- Thanks [tangmq](https://gitee.com/tangmq) for adding docker deployment service to PaddleOCR to support quick release of callable restful API services([#507](https://github.com/PaddlePaddle/PaddleOCR/pull/507)).
- Thanks [lijinhan](https://github.com/lijinhan) for adding Java springboot to PaddleOCR and call OCR hubserving interface to complete the use of OCR service deployment([#1027](https://github.com/PaddlePaddle/PaddleOCR/pull/1027)).
- Thanks [Evezerest](https://github.com/Evezerest), [ninetailskim](https://github.com/ninetailskim), [edencfc](https://github.com/edencfc), [BeyondYourself](https://github.com/BeyondYourself), [1084667371](https://github.com/1084667371)  for contributing  complete code of [PPOCRLabel](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/PPOCRLabel/README_ch.md).

### 1.3 Code And Document Optimization

- Thanks [zhangxin](https://github.com/ZhangXinNan)([Blog](https://blog.csdn.net/sdlypyzq)) for contributing new visualization methods and adding .gitgnore, handling the problem of manually setting the PYTHONPATH environment variable([#210](https://github.com/PaddlePaddle/PaddleOCR/pull/210)).
- Thanks [lyl120117](https://github.com/lyl120117) for contributing code to print network structure([#304](https://github.com/PaddlePaddle/PaddleOCR/pull/304)).
- Thanks [BeyondYourself](https://github.com/BeyondYourself) for making a lot of great suggestions for PaddleOCR and simplifying some code styles of paddleocr([so many commits)](https://github.com/PaddlePaddle/PaddleOCR/commits?author=BeyondYourself).
- Thanks [Khanh Tran](https://github.com/xxxpsyduck) and [Karl Horky](https://github.com/karlhorky) for contributing modifing English documents.

### 1.4 Multilingual Corpus

- Thanks [xiangyubo](https://github.com/xiangyubo)  for contributing handwritting Chinese OCR dataset([#321](https://github.com/PaddlePaddle/PaddleOCR/pull/321)).
- Thanks [Mejans](https://github.com/Mejans) for contributing dictionary and corpus of the new language Occitan to PaddleOCR([#954](https://github.com/PaddlePaddle/PaddleOCR/pull/954)).

## 2. Contribution Illustrating

### 2.1 New Function Class

PaddleOCR welcomes community contributions to various services, deployment examples and software applications with paddleOCR as the core. Certified community contributions will be added to the above community contribution table to increase exposure for the majority of developers, which is also the glory of PaddleOCR, including:

- Project form: the project code certified by the official community shall have good specifications and structure, and shall be equipped with a detailed README.md, which describes how to use the project. Through add a line 'paddleocr' to the requirements.txt, which can be automatically included in the usedby of paddleocr.

- Integration method: if it is an update to the existing PaddleOCR tool, it will be integrated into the main repo. If a new function is expanded for paddleocr, please contact the official personnel first to confirm whether the project is integrated into the master repo, *even if the new function is not integrated into the master repo, we will also increase the exposure of your personal project in the way of community contribution.* 


### 2.2 Code Optimization

If you encounter code bugs and unexpected functions when using PaddleOCR, you can contribute your modifications to PaddleOCR, including:

- Python code specifications are available for reference [Appendix 1:Python code specifications](./code_and_doc_en.md/#Appendix1).

- Before submitting the code, please confirm again and again that no new bugs will be introduced, and describe the optimization points in the PR. If the PR solves an issue, please connect to the issue in the PR. All PR shall comply with the requirements in Appendix [3.2.10 Some conventions for submitting code.](./code_and_doc_en.md/#Some conventions for submitting code)

- Please refer to the below before submitting. If you are not familiar with the git submission process, you can also refer to Section 3.2 of  [Appendix 3: description of Pull Request](./code_and_doc_en.md/#Appendix3).If you are not familiar with the git submission process, you can also refer to Section 3.2 of Appendix 3.

**Finally, please add the label Third Party in the title of PR and @ Everest in the description , PR with this label will be treated with high priority`[third-part]`.**

### 2.3 Document Optimization

If you encounter problems such as unclear document description, missing description and invalid link when using PaddleOCR, you can contribute your modifications to PaddleOCR. For document writing specifications, please refer to [Appendix 2: document specifications](./code_and_doc_en.md/#Appendix2). **Finally, please add the label Third Party in the title of PR and @ Everest in the description , PR with this label will be treated with high priority`[third-party].**

## 3. More Contribution Opportunities

We encourage developers to use PaddleOCR to realize their ideas. At the same time, we also list some valuable development directions after analysis, which are collected in the regular season of community projects as a whole.

## 4. Contact Us

We very much welcome developers to contact us before they intend to contribute code, documents, corpus and other contents to PaddleOCR, which can greatly reduce the communication cost in the PR process. At the same time, if you find some ideas difficult to realize personally, we can also recruit like-minded developers for the project in the form of SIG. Projects funded through SIG channels will receive deep R &amp; D support and operational resources (such as official account publicity, live broadcast lessons, etc.).

Our recommended contribution process is:

- By adding the `[Third Party]` mark in the topic of GitHub issue, explain the problems encountered (and the ideas to solve) or the functions to be expanded, and wait for the reply of the person on duty. For example, ` [Third Party] contributes IOS examples to PaddleOCR`.
- After communicating with us and confirming that the technical scheme or bugs and optimization points are correct, add functions or modify them accordingly, and the codes and documents shall comply with relevant specifications.
- PR links to the above issue and waits for review.

## 5. Thanks And Follow-Up

  - After the code is combined, the information will be updated in the first section of this document. The default link is GitHub name and home page. If you need to change the home page, you can also contact us.
  - New important function classes will be advertised in the user group and enjoy the honor of the open source community.
  - **If you have a PaddleOCR based project that does not appear in the above list, follow `4. Contact Us` .**
