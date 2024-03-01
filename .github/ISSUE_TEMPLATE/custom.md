---
name: Issue template
about: Template for reporting code errors.
title: ''
labels: ''
assignees: ''

---
body:
- type: checkboxes
  id: cat-preferences
  attributes:
    label: What kinds of cats do you like?
    description: You may select more than one.
    options:
      - label: Orange cat (required. Everyone likes orange cats.)
        required: true
      - label: **Black cat**


Please provide the following information to help us quickly identify and resolve the problem:

- **System Environment:** [e.g., Windows 10, macOS Catalina, Ubuntu 20.04]
- **Version:** [e.g., Paddle 2.0, PaddleOCR 2.1.0]
- **Related Components:** [e.g., PaddlePaddle, PaddleOCR]
- **Command Code:** [e.g., The command you used that resulted in the error]
- **Complete Error Message:** [The full error message you encountered]

Please avoid including images in your issue description if possible.

