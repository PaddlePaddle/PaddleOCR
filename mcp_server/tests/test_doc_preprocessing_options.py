from paddleocr_mcp.pipelines import OCRHandler, PPStructureV3Handler


def test_ocr_transforms_doc_preprocessing_options_per_call():
    handler = object.__new__(OCRHandler)

    local_kwargs = handler._transform_local_kwargs(
        {
            "use_doc_unwarping": True,
            "use_doc_orientation_classify": True,
        }
    )
    service_kwargs = handler._transform_service_kwargs(
        {
            "use_doc_unwarping": True,
            "use_doc_orientation_classify": True,
        }
    )

    assert local_kwargs == {
        "use_doc_unwarping": True,
        "use_doc_orientation_classify": True,
    }
    assert service_kwargs == {
        "useDocUnwarping": True,
        "useDocOrientationClassify": True,
    }


def test_layout_parsing_transforms_doc_preprocessing_options_per_call():
    handler = object.__new__(PPStructureV3Handler)
    handler._ppocr_source = "self_hosted"

    local_kwargs = handler._transform_local_kwargs(
        {
            "use_doc_unwarping": True,
            "use_doc_orientation_classify": False,
        }
    )
    service_kwargs = handler._transform_service_kwargs(
        {
            "use_doc_unwarping": True,
            "use_doc_orientation_classify": False,
        }
    )

    assert local_kwargs == {
        "use_doc_unwarping": True,
        "use_doc_orientation_classify": False,
    }
    assert service_kwargs["useDocUnwarping"] is True
    assert service_kwargs["useDocOrientationClassify"] is False


def test_doc_preprocessing_options_default_to_false_per_call():
    handler = object.__new__(OCRHandler)

    assert handler._transform_local_kwargs({}) == {
        "use_doc_unwarping": False,
        "use_doc_orientation_classify": False,
    }
    assert handler._transform_service_kwargs({}) == {
        "useDocUnwarping": False,
        "useDocOrientationClassify": False,
    }
