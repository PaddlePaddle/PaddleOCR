# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import types

def apply_langchain_shim():
    """
    A compatibility shim for LangChain to handle breaking changes in newer versions.
    Specifically addresses the removal of 'langchain.docstore'.
    """
    # Check if langchain is installed
    try:
        import langchain
    except ImportError:
        return

    # Ensure langchain is treated as a package if it's a dummy module
    if not hasattr(langchain, "__path__"):
        langchain.__path__ = []

    # Helper to create shim modules
    def create_shim(name, parent, attr):
        if not hasattr(parent, attr):
            mod = types.ModuleType(name)
            if not hasattr(mod, "__path__"):
                mod.__path__ = []
            sys.modules[name] = mod
            setattr(parent, attr, mod)
        return getattr(parent, attr)

    # Shim for docstore and document
    docstore = create_shim("langchain.docstore", langchain, "docstore")
    document = create_shim("langchain.docstore.document", docstore, "document")
    
    if not hasattr(document, "Document"):
        try:
            from langchain_core.documents import Document as RealDocument
            document.Document = RealDocument
        except ImportError:
            class MockDocument:
                def __init__(self, page_content, metadata=None):
                    self.page_content = page_content
                    self.metadata = metadata or {}
            document.Document = MockDocument

    # Shim for text_splitter
    text_splitter = create_shim("langchain.text_splitter", langchain, "text_splitter")
    if not hasattr(text_splitter, "RecursiveCharacterTextSplitter"):
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter as RealSplitter
            text_splitter.RecursiveCharacterTextSplitter = RealSplitter
        except ImportError:
            class MockSplitter:
                def __init__(self, *args, **kwargs): pass
            text_splitter.RecursiveCharacterTextSplitter = MockSplitter
