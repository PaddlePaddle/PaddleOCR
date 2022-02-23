import dill
dill.settings['recurse'] = True

def get_fun_with_strftime():
    def fun_with_strftime():
        import datetime
        return datetime.datetime.strptime("04-01-1943", "%d-%m-%Y").strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    return fun_with_strftime


def get_fun_with_strftime2():
    import datetime
    return datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')


def test_doc_dill_issue_219():
    back_fn = dill.loads(dill.dumps(get_fun_with_strftime()))
    assert back_fn() == "1943-01-04 00:00:00"
    dupl = dill.loads(dill.dumps(get_fun_with_strftime2))
    assert dupl() == get_fun_with_strftime2()


def get_fun_with_internal_import():
    def fun_with_import():
        import re
        return re.compile("$")
    return fun_with_import


def test_method_with_internal_import_should_work():
    import re
    back_fn = dill.loads(dill.dumps(get_fun_with_internal_import()))
    import inspect
    if hasattr(inspect, 'getclosurevars'):
        vars = inspect.getclosurevars(back_fn)
        assert vars.globals == {}
        assert vars.nonlocals == {}
    assert back_fn() == re.compile("$")
    assert "__builtins__" in back_fn.__globals__


if __name__ == "__main__":
    test_doc_dill_issue_219()
    test_method_with_internal_import_should_work()
