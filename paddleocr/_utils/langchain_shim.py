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
