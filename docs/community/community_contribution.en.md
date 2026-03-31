---
comments: true
typora-copy-images-to: images
---

# Community Contribution

Thank you for your long-term support and attention to PaddleOCR. Building a professional, harmonious, and mutually helpful open-source community with developers is the goal of PaddleOCR. This document showcases existing community contributions, explains various types of contributions, and describes new opportunities and processes, hoping to make the contribution process more efficient and the path clearer.

PaddleOCR hopes to help every developer with a dream realize their ideas through the power of AI and enjoy the pleasure of creating value.

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20" />
</a>

---

## 1. Community Contributions

### 1.1 New Features for PaddleOCR

- Many thanks to [authorfu](https://github.com/authorfu) for contributing the Android demo ([#340](https://github.com/PaddlePaddle/PaddleOCR/pull/340)) and [xxlyu-2046](https://github.com/xxlyu-2046) for contributing the iOS demo code ([#325](https://github.com/PaddlePaddle/PaddleOCR/pull/325)).
- Many thanks to [tangmq](https://gitee.com/tangmq) for adding Docker-based deployment services to PaddleOCR, supporting quick publishing of callable RESTful API services ([#507](https://github.com/PaddlePaddle/PaddleOCR/pull/507)).
- Many thanks to [lijinhan](https://github.com/lijinhan) for adding Java SpringBoot integration with the OCR HubServing interface for OCR service deployment ([#1027](https://github.com/PaddlePaddle/PaddleOCR/pull/1027)).
- Many thanks to [Evezerest](https://github.com/Evezerest), [ninetailskim](https://github.com/ninetailskim), [edencfc](https://github.com/edencfc), [BeyondYourself](https://github.com/BeyondYourself), and [1084667371](https://github.com/1084667371) for contributing the complete code for [PPOCRLabel](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/PPOCRLabel/README_ch.md).
- Many thanks to [bupt906](https://github.com/bupt906) for contributing the MicroNet architecture code ([#5251](https://github.com/PaddlePaddle/PaddleOCR/pull/5251)) and the OneCycle learning rate strategy code ([#5252](https://github.com/PaddlePaddle/PaddleOCR/pull/5252)).

### 1.2 Bug Fixes

- Many thanks to [zhangxin](https://github.com/ZhangXinNan) ([Blog](https://blog.csdn.net/sdlypyzq)) for contributing a new visualization method, adding .gitignore, and fixing the issue of manually setting the PYTHONPATH environment variable ([#210](https://github.com/PaddlePaddle/PaddleOCR/pull/210)).
- Many thanks to [lyl120117](https://github.com/lyl120117) for contributing the code for printing network structures ([#304](https://github.com/PaddlePaddle/PaddleOCR/pull/304)).
- Many thanks to [BeyondYourself](https://github.com/BeyondYourself) for providing many excellent suggestions and simplifying some of PaddleOCR's code style ([so many commits](https://github.com/PaddlePaddle/PaddleOCR/commits?author=BeyondYourself)).

### 1.3 Documentation Improvements and Translations

- Many thanks to **[RangeKing](https://github.com/RangeKing), [HustBestCat](https://github.com/HustBestCat), [v3fc](https://github.com/v3fc), [1084667371](https://github.com/1084667371)** for contributing the translation of the [English version of "Erta OCR" notebook e-book](https://github.com/PaddleOCR-Community/Dive-into-OCR/tree/main/notebook_en).
- Many thanks to [thunderstudying](https://github.com/thunderstudying), [RangeKing](https://github.com/RangeKing), [livingbody](https://github.com/livingbody), [WZMIAOMIAO](https://github.com/WZMIAOMIAO), and [haigang1975](https://github.com/haigang1975) for supplementing multiple English markdown documents.
- Many thanks to **[fanruinet](https://github.com/fanruinet)** for polishing and fixing 35 English documents ([#5205](https://github.com/PaddlePaddle/PaddleOCR/pull/5205)).
- Many thanks to [Khanh Tran](https://github.com/xxxpsyduck) and [Karl Horky](https://github.com/karlhorky) for contributing modifications to English documentation.

### 1.4 Multilingual Corpora

- Many thanks to [xiangyubo](https://github.com/xiangyubo) for contributing a handwritten Chinese OCR dataset ([#321](https://github.com/PaddlePaddle/PaddleOCR/pull/321)).
- Many thanks to [Mejans](https://github.com/Mejans) for adding a new Occitan language dictionary and corpus to PaddleOCR ([#954](https://github.com/PaddlePaddle/PaddleOCR/pull/954)).

## 2. Contribution Guidelines

### 2.1 New Features

PaddleOCR warmly welcomes community contributions of various services, deployment examples, and software applications built around PaddleOCR. Certified community contributions will be added to the community contribution table above, providing increased visibility for developers and bringing honor to PaddleOCR:

- Project format: Officially certified community project code should follow good standards and structure, and should also include a detailed README.md explaining how to use the project. Adding `paddleocr` to your requirements.txt file will automatically include your project in PaddleOCR's "used by" list.

- Merge process: If it is an update or upgrade to an existing PaddleOCR tool, it will be merged into the main repo. If it extends new functionality for PaddleOCR, please contact the official team first to confirm whether the project should be merged into the main repo. *Even if the new functionality is not merged into the main repo, we will still increase visibility for your personal project as a community contribution.*

### 2.2 Code Optimization

If you encounter code bugs or unexpected behavior while using PaddleOCR, you can contribute your fixes:

- For Python code standards, please refer to [Appendix 1: Python Code Specification](./code_and_doc.md#appendix-1python-code-specification).

- Please double-check before submitting code that no new bugs are introduced, and describe the optimization points in the PR. If the PR resolves a specific issue, please link to that issue in the PR. All PRs should follow the conventions in [3.2.10 Some Conventions For Submitting Code](./code_and_doc.md#3210-some-conventions-for-submitting-code).

- Please refer to [Appendix 3: Pull Request Description](./code_and_doc.md#appendix-3-pull-request-description) before submitting. If you are not familiar with the git submission process, you can also refer to section 3.2 of Appendix 3.

### 2.3 Documentation Improvements

If you encounter unclear descriptions, missing content, or broken links while using PaddleOCR, you can contribute your fixes. For documentation writing standards, please refer to [Appendix 2: Document Specification](./code_and_doc.md#appendix-2-document-specification).

## 3. More Contribution Opportunities

We strongly encourage developers to use PaddleOCR to implement their own ideas. We also list some valuable extension directions that have been analyzed and are collected in the community project regular competition.

## 4. Contact Us

We warmly welcome developers to contact us before contributing code, documentation, corpora, or other content to PaddleOCR. This can greatly reduce communication costs during the PR process. Additionally, if you find certain ideas difficult to implement individually, we can recruit like-minded developers through SIG (Special Interest Group) to collaborate on the project. Projects contributed through the SIG channel will receive in-depth R&D support and operational resources (such as public account promotion, live courses, etc.).

Our recommended contribution process is:

- Add the `[third-party]` tag to the GitHub issue title, describe the problem encountered (and your proposed solution) or the functionality you want to extend, and wait for a response from the duty personnel. For example: `[third-party] Contributing an iOS example for PaddleOCR`.
- After confirming the technical solution or verifying the bug/optimization through communication with us, proceed with adding the new functionality or making the relevant changes. Code and documentation should follow the relevant standards.
- Link the PR to the above issue and wait for review.

## 5. Acknowledgments and Follow-up

- After the code is merged, information will be updated in the first section of this document. By default, links point to the GitHub username and homepage. If you need to change the homepage, you can also contact us.
- For major new features, announcements will be made in user groups to share the honor of open-source community contributions.
- **If you have a project based on PaddleOCR that is not listed above, please follow the steps in "4. Contact Us" to reach out to us.**
