# Handwritten OCR dataset
Here we have sorted out the commonly used handwritten OCR dataset datasets, which are being updated continuously. We welcome you to contribute datasets ~
- [Institute of automation, Chinese Academy of Sciences - handwritten Chinese dataset](#Institute of automation, Chinese Academy of Sciences - handwritten Chinese dataset)
- [NIST handwritten single character dataset - English](#NIST handwritten single character dataset - English)

<a name="Institute of automation, Chinese Academy of Sciences - handwritten Chinese dataset"></a>
## Institute of automation, Chinese Academy of Sciences - handwritten Chinese dataset
- **Data source**:http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html
- **Data introduction**:
    * It includes online and offline handwritten data,`HWDB1.0~1.2` has totally 3895135 handwritten single character samples, which belong to 7356 categories (7185 Chinese characters and 171 English letters, numbers and symbols);`HWDB2.0~2.2` has totally 5091 pages of images, which are divided into 52230 text lines and 1349414 words. All text and text samples are stored as grayscale images. Some sample words are shown below.

        ![](../datasets/CASIA_0.jpg)

- **Download address**:http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html
- **使用建议**:Data for single character, white background, can form a large number of text lines for training. White background can be processed into transparent state, which is convenient to add various backgrounds. For the case of semantic needs, it is suggested to extract single character from real corpus to form text lines.


<a name="NIST handwritten single character dataset - English"></a>
## NIST handwritten single character dataset - English(NIST Handprinted Forms and Characters Database)

- **Data source**: [https://www.nist.gov/srd/nist-special-database-19](https://www.nist.gov/srd/nist-special-database-19)

- **Data introduction**: NIST19 dataset is suitable for handwritten document and character recognition model training. It is extracted from the handwritten sample form of 3600 authors and contains 810000 character images in total. Nine of them are shown below.

    ![](../datasets/nist_demo.png)


- **Download address**: [https://www.nist.gov/srd/nist-special-database-19](https://www.nist.gov/srd/nist-special-database-19)
