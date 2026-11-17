import scsp


def test_core_module_loads():
    assert scsp.version() == scsp.__version__
    assert "c++2017" in scsp.build_info()
    assert scsp.hardware_threads() >= 1


def test_std_thread_runtime():
    n = 123_457
    for threads in (1, 2, 4, 7):
        assert scsp.parallel_sum(n, threads) == n * (n + 1) // 2
