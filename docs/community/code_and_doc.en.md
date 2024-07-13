---
comments: true
typora-copy-images-to: images
---

# Appendix

  This appendix contains python, document specifications and Pull Request process.

## Appendix 1：Python Code Specification

The Python code of PaddleOCR follows [PEP8 Specification]( https://www.python.org/dev/peps/pep-0008/ ), some of the key concerns include the following

- Space

    - Spaces should be added after commas, semicolons, colons, not before them

      ```python linenums="1"
      # true:
      print(x, y)

      # false:
      print(x , y)
      ```

    - When specifying a keyword parameter or default parameter value in a function, do not use spaces on both sides of it

      ```python linenums="1"
      # true:
      def complex(real, imag=0.0)
      # false:
      def complex(real, imag = 0.0)
      ```

- comment

    - Inline comments: inline comments are indicated by the` # `sign. Two spaces should be left between code and` # `, and one space should be left between` # `and comments, for example

      ```python linenums="1"
      x = x + 1  # Compensate for border
      ```

    - Functions and methods: The definition of each function should include the following:

        - Function description: Utility, input and output of function
        - Args: Name and description of each parameter
        - Returns: The meaning and type of the return value

        ```python linenums="1"
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

## Appendix 2: Document Specification

### 2.1 Overall Description

- Document Location: If you add new features to your original Markdown file, please **Do not re-create** a new file. If you don't know where to add it, you can first PR the code and then ask the official in commit.

- New Markdown Document Name: Describe the content of the document in English, typically a combination of lowercase letters and underscores, such as `add_New_Algorithm.md`

- New Markdown Document Format: Catalog - Body - FAQ

  > The directory generation method can use [this site](https://ecotrust-canada.github.io/markdown-toc/ ) Automatically extract directories after copying MD contents, and then add `

- English and Chinese: Any changes or additions to the document need to be made in both Chinese and English documents.

### 2.2 Format Specification

- Title format: The document title format follows the format of: Arabic decimal point combination-space-title (for example, `2.1 XXXX`, `2.XXXX`)

- Code block: Displays code in code block format that needs to be run, describing the meaning of command parameters before the code block. for example:

  > Pipeline of detection + direction Classify + recognition: Vertical text can be recognized after set direction classifier parameters`--use_angle_cls true`.
  >
  > ```bash linenums="1"
  > paddleocr --image_dir ./imgs/11.jpg --use_angle_cls true
  > ```

- Variable Rrferences: If code variables or command parameters are referenced in line, they need to be represented in line code, for example, above `--use_angle_cls true` with one space in front and one space in back

- Uniform naming: e.g. PP-OCRv2, PP-OCR mobile, `paddleocr` whl package, PPOCRLabel, Paddle Lite, etc.

- Supplementary notes: Supplementary notes by reference format `>`.

- Picture: If a picture is added to the description document, specify the naming of the picture (describing its content) and add the picture under `doc/`.

- Title: Capitalize the first letter of each word in the title.

## Appendix 3: Pull Request Description

### 3.1 PaddleOCR Branch Description

PaddleOCR will maintain two branches in the future, one for each:

- release/x.x family branch: stable release version branch, also the default branch. PaddleOCR releases a new release branch based on feature updates and adapts to the release version of Paddle. As versions iterate, more and more release/x.x family branches are maintained by default with the latest version of the release branch.
- dygraph branch: For the development branch, adapts the dygraph version of the Paddle dynamic graph to primarily develop new functionality. If you need to redevelop, choose the dygraph branch. To ensure that the dygraph branch pulls out the release/x.x branch when needed, the code for the dygraph branch can only use the valid API in the latest release branch of Paddle. That is, if a new API has been developed in the Paddle dygraph branch but has not yet appeared in the release branch code, do not use it in Paddle OCR. In addition, performance optimization, parameter tuning, policy updates that do not involve API can be developed normally.

The historical branch of PaddleOCR will no longer be maintained in the future. These branches will continue to be maintained, considering that some of you may still be using them:

Develop branch: This branch was used for the development and testing of static diagrams and is currently compatible with version >=1.7. If you have special needs, you can also use this branch to accommodate older versions of Paddle, but you won't update your code until you fix the bug.

PaddleOCR welcomes you to actively contribute code to repo. Here are some basic processes for contributing code.

### 3.2 PaddleOCR Code Submission Process And Specification

If you are familiar with Git use, you can jump directly to [Some Conventions For Submitting Code in 3.2.10](#Some_conventions_for_submitting_code)

#### 3.2.1 Create Your `Remote Repo`

In PaddleOCR [GitHub Home]( https://github.com/PaddlePaddle/PaddleOCR ) Click the `Fork` button in the upper left corner to create a `remote repo`in your personal directory, such as `https://github.com/ {your_name}/PaddleOCR`.

  ![banner](./images/banner.png)

Clone `Remote repo`

  ```bash linenums="1"
  # pull code of develop branch
  git clone https://github.com/{your_name}/PaddleOCR.git -b dygraph
  cd PaddleOCR
  ```

Clone failures are mostly due to network reasons, try again later or configure the proxy

#### 3.2.2 Login And Connect Using Token

Start by viewing the information for the current `remote repo`.

```bash linenums="1"
git remote -v
# origin    https://github.com/{your_name}/PaddleOCR.git (fetch)
# origin    https://github.com/{your_name}/PaddleOCR.git (push)
```

Only the information of the clone `remote repo`, i.e. the PaddleOCR under your username, is available. Due to the change in Github's login method, you need to reconfigure the `remote repo` address by means of a Token. The token is generated as follows:

1. Find Personal Access Tokens: Click on your avatar in the upper right corner of the Github page and choose Settings --> Developer settings --> Personal access tokens,

2. Click Generate new token: Fill in the token name in Note, such as 'paddle'. In Select scopes, select repo (required), admin:repo_hook, delete_repo, etc. You can check them according to your needs. Then click Generate token to generate the token, and finally copy the generated token.

Delete the original origin configuration

```bash linenums="1"
git remote rm origin
```

Change the remote branch to `https://oauth2:{token}@github.com/{your_name}/PaddleOCR.git`. For example, if the token value is 12345 and your user name is PPOCR, run the following command

```bash linenums="1"
git remote add origin https://oauth2:12345@github.com/PPOCR/PaddleOCR.git
```

This establishes a connection to our own `remote repo`. Next we create a remote host of the original PaddleOCR repo, named upstream.

```bash linenums="1"
git remote add upstream https://github.com/PaddlePaddle/PaddleOCR.git
```

Use `git remote -v` to view current `remote warehouse` information, output as follows, found to include two origin and two upstream of `remote repo` .

```bash linenums="1"
origin    https://github.com/{your_name}/PaddleOCR.git (fetch)
origin    https://github.com/{your_name}/PaddleOCR.git (push)
upstream    https://github.com/PaddlePaddle/PaddleOCR.git (fetch)
upstream    https://github.com/PaddlePaddle/PaddleOCR.git (push)
```

This is mainly to keep the local repository up to date when subsequent pull request (PR) submissions are made.

#### 3.2.3 Create Local Branch

First get the latest code of upstream, then create a new_branch branch based on the dygraph of the upstream repo (upstream).

```bash linenums="1"
git fetch upstream
git checkout -b new_branch upstream/dygraph
```

> If for a newly forked PaddleOCR project, the user's remote repo (origin) has the same branch updates as the upstream repository (upstream), you can also create a new local branch based on the default branch of the origin repo or a specified branch with the following command
>
> ```bash linenums="1"
> # Create new_branch branch on user remote repo (origin) based on develop branch
> git checkout -b new_branch origin/develop
> # Create new_branch branch based on upstream remote repo develop branch
> # If you need to create a new branch from upstream,
> # you need to first use git fetch upstream to get upstream code
> git checkout -b new_branch upstream/develop
> ```

The final switch to the new branch is displayed with the following output information.

Branch new_branch set up to track remote branch develop from upstream.
Switched to a new branch 'new_branch'

After switching branches, file changes can be made on this branch

#### 3.2.4 Use Pre-Commit Hook

Paddle developers use the pre-commit tool to manage Git pre-submit hooks. It helps us format the source code (C++, Python) and automatically check for basic things (such as having only one EOL per file, not adding large files to Git) before committing it.

The pre-commit test is part of the unit test in Travis-CI. PR that does not satisfy the hook cannot be submitted to PaddleOCR. Install it first and run it in the current directory：

```bash linenums="1"
pip install pre-commit
pre-commit install
```

> 1. Paddle uses clang-format to adjust the C/C++ source code format. Make sure the `clang-format` version is above 3.8.
>
> 2. Yapf installed through pip install pre-commit is slightly different from conda install-c conda-forge pre-commit, and PaddleOCR developers use `pip install pre-commit`.

#### 3.2.5 Modify And Submit Code

If you make some changes on `README.Md` on PaddleOCR, you can view the changed file through `git status`, and then add the changed file using `git add`。

```bash linenums="1"
git status # View change files
git add README.md
pre-commit
```

Repeat these steps until the pre-comit format check does not error. As shown below.

![img](./images/precommit_pass.png)

Use the following command to complete the submission.

```bash linenums="1"
git commit -m "your commit info"
```

#### 3.2.6 Keep Local Repo Up To Date

Get the latest code for upstream and update the current branch. Here the upstream comes from section 2.2, `Connecting to a remote repo`.

```bash linenums="1"
git fetch upstream
# If you want to commit to another branch, you need to pull code from another branch of upstream, here is develop
git pull upstream develop
```

#### 3.2.7 Push To Remote Repo

```bash linenums="1"
git push origin new_branch
```

#### 3.2.7 Submit Pull Request

Click the new pull request to select the local branch and the target branch, as shown in the following figure. In the description of PR, fill in the functions completed by the PR. Next, wait for review, and if you need to modify something, update the corresponding branch in origin with the steps above.

![banner](./images/pr.png)

#### 3.2.8 Sign CLA Agreement And Pass Unit Tests

Signing the CLA When submitting a Pull Request to PaddlePaddle for the first time, you need to sign a CLA (Contributor License Agreement) agreement to ensure that your code can be incorporated as follows:

1. Please check the Check section in PR, find the license/cla, and click on the right detail to enter the CLA website

2. Click Sign in with GitHub to agree on the CLA website and when clicked, it will jump back to your Pull Request page

#### 3.2.9 Delete Branch

- Remove remote branch

  After PR is merged into the main repo, we can delete the branch of the remote repofrom the PR page.
  You can also use `git push origin:branch name` to delete remote branches, such as:

  ```bash linenums="1"
  git push origin :new_branch
  ```

- Delete local branch

  ```bash linenums="1"
  # Switch to the development branch, otherwise the current branch cannot be deleted
  git checkout develop

  # Delete new_ Branch Branch
  git branch -D new_branch
  ```

#### 3.2.10 Some Conventions For Submitting Code

In order for official maintainers to better focus on the code itself when reviewing it, please follow the following conventions each time you submit your code:

1）Please ensure that the unit tests in Travis-CI pass smoothly. If not, indicate that there is a problem with the submitted code, and the official maintainer generally does not review it.

2）Before submitting a Pull Request.

- Note the number of commits.

  Reason: If you only modify one file and submit more than a dozen commits, each commit will only make a few modifications, which can be very confusing to the reviewer. The reviewer needs to look at each commit individually to see what changes have been made, and does not exclude the fact that changes between commits overlap each other.

  Suggestion: Keep as few commits as possible each time you submit, and supplement your last commit with git commit --amend. For multiple commits that have been Push to a remote warehouse, you can refer to [squash commits after push](https://stackoverflow.com/questions/5667884/how-to-squash-commits-in-git-after-they-have-been-pushed ).

- Note the name of each commit: it should reflect the content of the current commit, not be too arbitrary.

3） If you have solved a problem, add in the first comment box of the Pull Request:fix #issue_number，This will automatically close the corresponding Issue when the Pull Request is merged. Key words include:close, closes, closed, fix, fixes, fixed, resolve, resolves, resolved,please choose the right vocabulary. Detailed reference [Closing issues via commit messages](https://help.github.com/articles/closing-issues-via-commit-messages).

In addition, in response to the reviewer's comments, you are requested to abide by the following conventions:

1） Each review comment from an official maintainer would like a response, which would better enhance the contribution of the open source community.

- If you agree to the review opinion and modify it accordingly, give a simple Done.
- If you disagree with the review, please give your own reasons for refuting.

2）If there are many reviews:

- Please give an overview of the changes.
- Please reply with `start a review', not directly. The reason is that each reply sends an e-mail message, which can cause a mail disaster.
