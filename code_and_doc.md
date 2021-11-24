# 附录

本附录包含了Python、文档规范以及Pull Request流程，请各位开发者遵循相关内容

- [附录1：Python代码规范](#附录1)

- [附录2：文档规范](#附录2)

- [附录3：Pull Request说明](#附录3)

<a name="附录1"></a>

## 附录1：Python代码规范

PaddleOCR的Python代码遵循 [PEP8规范](https://www.python.org/dev/peps/pep-0008/)，其中一些关注的重点包括如下内容

- 空格 

  - 空格应该加在逗号、分号、冒号前，而非他们的后面

    ```python
    # 正确：
    print(x, y)
    
    # 错误：
    print(x , y)
    ```

  - 在函数中指定关键字参数或默认参数值时, 不要在其两侧使用空格

    ```python
    # 正确：
    def complex(real, imag=0.0)
    # 错误：
    def complex(real, imag = 0.0)
    ```

- 注释

  - 行内注释：行内注释使用 `#` 号表示，在代码与 `#` 之间需要空两个空格， `#`  与注释之间应当空一个空格，例如

    ```python
    x = x + 1  # Compensate for border
    ```

  - 函数和方法：每个函数的定义后的描述应该包括以下内容：

    - 函数描述：函数的作用，输入输出的

    - Args：每个参数的名字以及对该参数的描述
    - Returns：返回值的含义和类型

    ```python
    def fetch_bigtable_rows(big_table, keys, other_silly_variable=None):
        """Fetches rows from a Bigtable.
    
        Retrieves rows pertaining to the given keys from the Table instance
        represented by big_table.  Silly things may happen if
        other_silly_variable is not None.
    
        Args:
            big_table: An open Bigtable Table instance.
            keys: A sequence of strings representing the key of each table row
                to fetch.
            other_silly_variable: Another optional variable, that has a much
                longer name than the other args, and which does nothing.
    
        Returns:
            A dict mapping keys to the corresponding table row data
            fetched. Each row is represented as a tuple of strings. For
            example:
    
            {'Serak': ('Rigel VII', 'Preparer'),
             'Zim': ('Irk', 'Invader'),
             'Lrrr': ('Omicron Persei 8', 'Emperor')}
    
            If a key from the keys argument is missing from the dictionary,
            then that row was not found in the table.
        """
        pass
    ```

<a name="附录2"></a>

## 附录2：文档规范

### 2.1 总体说明

- 文档位置：如果您增加的新功能可以补充在原有的Markdown文件中，请**不要重新新建**一个文件。如果您对添加的位置不清楚，可以先PR代码，然后在commit中询问官方人员。

- 新增Markdown文档名称：使用英文描述文档内容，一般由小写字母与下划线组合而成，例如  `add_new_algorithm.md`

- 新增Markdown文档格式：目录 - 正文 - FAQ

  > 目录生成方法可以使用 [此网站](https://ecotrust-canada.github.io/markdown-toc/) 将md内容复制之后自动提取目录，然后在md文件的每个标题前添加 `<a name="XXXX"></a>` 

- 中英双语：任何对文档的改动或新增都需要分别在中文和英文文档上进行。

### 2.2 格式规范

- 标题格式：文档标题格式按照：阿拉伯数字小数点组合 - 空格 - 标题的格式（例如 `2.1 XXXX` ， `2. XXXX`）

- 代码块：通过代码块格式展示需要运行的代码，在代码块前描述命令参数的含义。例如：

  > 检测+方向分类器+识别全流程：设置方向分类器参数 `--use_angle_cls true` 后可对竖排文本进行识别。
  >
  > ```
  > paddleocr --image_dir ./imgs/11.jpg --use_angle_cls true
  > ```

- 变量引用：如果在行内引用到代码变量或命令参数，需要用行内代码表示，例如上方  `--use_angle_cls true` ，并在前后各空一格

- 补充说明：通过引用格式 `>` 补充说明，或对注意事项进行说明

- 图片：如果在说明文档中增加了图片，请规范图片的命名形式（描述图片内容），并将图片添加在 `doc/` 下

<a name="附录3"></a>

## 附录3：Pull Request说明

### 3.1 PaddleOCR分支说明

PaddleOCR未来将维护2种分支，分别为：

- release/x.x系列分支：为稳定的发行版本分支，会适时打tag发布版本，适配Paddle的release版本。当前最新的分支为release/2.0分支，是当前默认分支，适配Paddle v2.0.0。随着版本迭代，release/x.x系列分支会越来越多，默认维护最新版本的release分支，前1个版本分支会修复bug，其他的分支不再维护。
- dygraph分支：为开发分支，适配Paddle动态图的dygraph版本，主要用于开发新功能。如果有同学需要进行二次开发，请选择dygraph分支。为了保证dygraph分支能在需要的时候拉出release/x.x分支，dygraph分支的代码只能使用Paddle最新release分支中有效的api。也就是说，如果Paddle dygraph分支中开发了新的api，但尚未出现在release分支代码中，那么请不要在PaddleOCR中使用。除此之外，对于不涉及api的性能优化、参数调整、策略更新等，都可以正常进行开发。

PaddleOCR的历史分支，未来将不再维护。考虑到一些同学可能仍在使用，这些分支还会继续保留：

- develop分支：这个分支曾用于静态图的开发与测试，目前兼容>=1.7版本的Paddle。如果有特殊需求，要适配旧版本的Paddle，那还可以使用这个分支，但除了修复bug外不再更新代码。

PaddleOCR欢迎大家向repo中积极贡献代码，下面给出一些贡献代码的基本流程。

### 3.2 PaddleOCR代码提交流程与规范

> 如果你熟悉Git使用，可以直接跳转到 [3.2.10 提交代码的一些约定](#提交代码的一些约定)

#### 3.2.1 创建你的 `远程仓库`

- 在PaddleOCR的 [GitHub首页](https://github.com/PaddlePaddle/PaddleOCR)，点击左上角 `Fork`  按钮，在你的个人目录下创建 `远程仓库`，比如`https://github.com/{your_name}/PaddleOCR`。

![banner](/Users/zhulingfeng01/OCR/PaddleOCR/doc/banner.png)

- 将 `远程仓库` Clone到本地

```
# 拉取develop分支的代码
git clone https://github.com/{your_name}/PaddleOCR.git -b dygraph
cd PaddleOCR
```

> 多数情况下clone失败是由于网络原因，请稍后重试或配置代理

#### 3.2.2 和 `远程仓库` 建立连接

首先查看当前 `远程仓库` 的信息。

```
git remote -v
# origin    https://github.com/{your_name}/PaddleOCR.git (fetch)
# origin    https://github.com/{your_name}/PaddleOCR.git (push)
```

只有clone的 `远程仓库` 的信息，也就是自己用户名下的 PaddleOCR，接下来我们创建一个原始 PaddleOCR 仓库的远程主机，命名为 upstream。

```
git remote add upstream https://github.com/PaddlePaddle/PaddleOCR.git
```

使用 `git remote -v` 查看当前 `远程仓库` 的信息，输出如下，发现包括了origin和upstream 2个 `远程仓库` 。

```
origin    https://github.com/{your_name}/PaddleOCR.git (fetch)
origin    https://github.com/{your_name}/PaddleOCR.git (push)
upstream    https://github.com/PaddlePaddle/PaddleOCR.git (fetch)
upstream    https://github.com/PaddlePaddle/PaddleOCR.git (push)
```

这主要是为了后续在提交pull request(PR)时，始终保持本地仓库最新。

#### 3.2.3 创建本地分支

可以基于当前分支创建新的本地分支，命令如下。

```
git checkout -b new_branch
```

也可以基于远程或者上游的分支创建新的分支，命令如下。

```
# 基于用户远程仓库(origin)的develop创建new_branch分支
git checkout -b new_branch origin/develop
# 基于上游远程仓库(upstream)的develop创建new_branch分支
# 如果需要从upstream创建新的分支，需要首先使用git fetch upstream获取上游代码
git checkout -b new_branch upstream/develop
```

最终会显示切换到新的分支，输出信息如下

```
Branch new_branch set up to track remote branch develop from upstream.
Switched to a new branch 'new_branch'
```

#### 3.2.4 使用pre-commit勾子

Paddle 开发人员使用 pre-commit 工具来管理 Git 预提交钩子。 它可以帮助我们格式化源代码（C++，Python），在提交（commit）前自动检查一些基本事宜（如每个文件只有一个 EOL，Git 中不要添加大文件等）。

pre-commit测试是 Travis-CI 中单元测试的一部分，不满足钩子的 PR 不能被提交到 PaddleOCR，首先安装并在当前目录运行它：

```
pip install pre-commit
pre-commit install
```

 >  1. Paddle 使用 clang-format 来调整 C/C++ 源代码格式，请确保 `clang-format` 版本在 3.8 以上。
 >
 >  2. 通过pip install pre-commit和conda install -c conda-forge pre-commit安装的yapf稍有不同的，PaddleOCR 开发人员使用的是 `pip install pre-commit`。

#### 3.2.5 修改与提交代码

 假设对PaddleOCR的 `README.md` 做了一些修改，可以通过 `git status` 查看改动的文件，然后使用 `git add` 添加改动文件。

```
git status # 查看改动文件
git add README.md
pre-commit
```

重复上述步骤，直到pre-comit格式检查不报错。如下所示。

[![img](https://github.com/PaddlePaddle/PaddleClas/raw/release/2.3/docs/images/quick_start/community/003_precommit_pass.png)](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/images/quick_start/community/003_precommit_pass.png)

使用下面的命令完成提交。

```
git commit -m "your commit info"
```

#### 3.2.6 保持本地仓库最新

获取 upstream 的最新代码并更新当前分支。这里的upstream来自于2.2节的`和远程仓库建立连接`部分。

```
git fetch upstream
# 如果是希望提交到其他分支，则需要从upstream的其他分支pull代码，这里是develop
git pull upstream develop
```

#### 3.2.7 push到远程仓库

```
git push origin new_branch
```

#### 3.2.7 提交Pull Request

点击new pull request，选择本地分支和目标分支，如下图所示。在PR的描述说明中，填写该PR所完成的功能。接下来等待review，如果有需要修改的地方，参照上述步骤更新 origin 中的对应分支即可。

![banner](/Users/zhulingfeng01/OCR/PaddleOCR/doc/pr.png)

#### 3.2.8 签署CLA协议和通过单元测试

- 签署CLA 在首次向PaddlePaddle提交Pull Request时，您需要您签署一次CLA(Contributor License Agreement)协议，以保证您的代码可以被合入，具体签署方式如下：

  1. 请您查看PR中的Check部分，找到license/cla，并点击右侧detail，进入CLA网站

  2. 点击CLA网站中的“Sign in with GitHub to agree”,点击完成后将会跳转回您的Pull Request页面

#### 3.2.9 删除分支

- 删除远程分支

  在 PR 被 merge 进主仓库后，我们可以在 PR 的页面删除远程仓库的分支。

  也可以使用 `git push origin :分支名` 删除远程分支，如：

  ```
  git push origin :new_branch
  ```

- 删除本地分支

  ```
  # 切换到develop分支，否则无法删除当前分支
  git checkout develop
  
  # 删除new_branch分支
  git branch -D new_branch
  ```

<a name="提交代码的一些约定"></a>

#### 3.2.10 提交代码的一些约定

为了使官方维护人员在评审代码时更好地专注于代码本身，请您每次提交代码时，遵守以下约定：

1）请保证Travis-CI 中单元测试能顺利通过。如果没过，说明提交的代码存在问题，官方维护人员一般不做评审。

2）提交Pull Request前：

- 请注意commit的数量。

  原因：如果仅仅修改一个文件但提交了十几个commit，每个commit只做了少量的修改，这会给评审人带来很大困扰。评审人需要逐一查看每个commit才能知道做了哪些修改，且不排除commit之间的修改存在相互覆盖的情况。

  建议：每次提交时，保持尽量少的commit，可以通过git commit --amend补充上次的commit。对已经Push到远程仓库的多个commit，可以参考[squash commits after push](https://stackoverflow.com/questions/5667884/how-to-squash-commits-in-git-after-they-have-been-pushed)。

- 请注意每个commit的名称：应能反映当前commit的内容，不能太随意。


3）如果解决了某个Issue的问题，请在该Pull Request的第一个评论框中加上：fix #issue_number，这样当该Pull Request被合并后，会自动关闭对应的Issue。关键词包括：close, closes, closed, fix, fixes, fixed, resolve, resolves, resolved，请选择合适的词汇。详细可参考[Closing issues via commit messages](https://help.github.com/articles/closing-issues-via-commit-messages)。

此外，在回复评审人意见时，请您遵守以下约定：

1）官方维护人员的每一个review意见都希望得到回复，这样会更好地提升开源社区的贡献。

- 对评审意见同意且按其修改完的，给个简单的Done即可；
- 对评审意见不同意的，请给出您自己的反驳理由。

2）如果评审意见比较多:

- 请给出总体的修改情况。
- 请采用`start a review`进行回复，而非直接回复的方式。原因是每个回复都会发送一封邮件，会造成邮件灾难。