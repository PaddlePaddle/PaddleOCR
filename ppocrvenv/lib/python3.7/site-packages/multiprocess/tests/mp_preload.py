import multiprocessing as multiprocess

multiprocess.Lock()


def f():
    print("ok")


if __name__ == "__main__":
    ctx = multiprocess.get_context("forkserver")
    modname = "test.mp_preload"
    # Make sure it's importable
    __import__(modname)
    ctx.set_forkserver_preload([modname])
    proc = ctx.Process(target=f)
    proc.start()
    proc.join()
