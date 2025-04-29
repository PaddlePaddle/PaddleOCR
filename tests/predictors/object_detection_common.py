def check_result_item_keys(result_item):
    assert result_item.keys() == {
        "input_path",
        "page_index",
        "input_img",
        "boxes",
    }
